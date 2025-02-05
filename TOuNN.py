import numpy as np  # Ensure NumPy is installed using: pip install numpy
import jax.numpy as jnp
import jax
from jax import jit
import matplotlib.pyplot as plt
from matplotlib import colors
from FE_Solver import JAXSolver
from network import TopNet
from projections import applyFourierMap, applySymmetry, applyRotationalSymmetry, applyExtrusion
from jax.example_libraries import optimizers
from materialCoeffs import microStrs
import pickle


class TOuNN:
  def __init__(self, exampleName, mesh, material, nnSettings, symMap,
               fourierMap, rotationalSymmetry, extrusion):
    self.exampleName = exampleName
    self.FE = JAXSolver(mesh, material)  # Update solver to handle heat transfer
    self.xy = self.FE.mesh.elemCenters
    self.fourierMap = fourierMap
    if fourierMap['isOn']:
      nnSettings['inputDim'] = 2 * fourierMap['numTerms']
    else:
      nnSettings['inputDim'] = self.FE.mesh.ndim
    self.topNet = TopNet(nnSettings)

    self.symMap = symMap
    self.mstrData = microStrs
    self.rotationalSymmetry = rotationalSymmetry
    self.extrusion = extrusion

  def optimizeDesign(self, optParams, savedNet):
    convgHistory = {'epoch': [], 'vf': [], 'J': []}
    xyS = applySymmetry(self.xy, self.symMap)
    xyE = applyExtrusion(xyS, self.extrusion)
    xyR = applyRotationalSymmetry(xyE, self.rotationalSymmetry)
    xyF = applyFourierMap(xyR, self.fourierMap) if self.fourierMap['isOn'] else xyR
    penal = 1

    # Compute Effective Thermal Conductivity Instead of Stiffness Matrix
    def getThermalConductivity(vfracPow, mstr):
      """Compute effective thermal conductivity from polynomial model"""
      k = jnp.zeros((self.FE.mesh.numElems))
      for pw in range(mstr['order'] + 1):
        k = k.at[:].add(mstr['k'][str(pw)] * vfracPow[str(pw)])
      return k

    @jit
    def getThermalMatrix(mstrType, nn_rho):
      vfracPow = {str(pw): nn_rho ** pw for pw in range(self.mstrData['square']['order'] + 1)}
      k = jnp.zeros((self.FE.mesh.numElems))

      for mstrCtr, mstr in enumerate(self.mstrData):
        k_mstr = getThermalConductivity(vfracPow, self.mstrData[mstr])
        mstrPenal = mstrType[:, mstrCtr] ** penal
        k = k.at[:].add(jnp.einsum('i,i->i', mstrPenal, k_mstr))

      return k

    # Optimizer Setup
    opt_init, opt_update, get_params = optimizers.adam(optParams['learningRate'])
    opt_state = opt_init(self.topNet.wts)
    opt_update = jit(opt_update)

    if savedNet['isAvailable']:
      saved_params = pickle.load(open(savedNet['file'], "rb"))
      opt_state = optimizers.pack_optimizer_state(saved_params)

    # Forward Pass to Get Initial Loss Scaling Factor
    mstrType, density0 = self.topNet.forward(get_params(opt_state), xyF)
    k = getThermalMatrix(mstrType, 0.01 + density0)
    J0 = self.FE.objectiveHandle(k)  # Compute initial objective function

    def computeLoss(objective, constraints):
      """Compute total loss with constraints"""
      if optParams['lossMethod']['type'] == 'penalty':
        alpha = optParams['lossMethod']['alpha0'] + self.epoch * optParams['lossMethod']['delAlpha']
        loss = objective
        for c in constraints:
          loss += alpha * c ** 2
      elif optParams['lossMethod']['type'] == 'logBarrier':
        t = optParams['lossMethod']['t0'] * optParams['lossMethod']['mu'] ** self.epoch
        loss = objective
        for c in constraints:
          psi = -jnp.log(-c) / t if c < (-1 / t ** 2) else t * c - jnp.log(1 / t ** 2) / t + 1 / t
          loss += psi
      return loss

    def closure(nnwts):
      """Closure function to compute loss for optimization"""
      mstrType, density = self.topNet.forward(nnwts, xyF)
      volCons = (jnp.mean(density) / optParams['desiredVolumeFraction']) - 1.
      k = getThermalMatrix(mstrType, 0.01 + density)
      J = self.FE.objectiveHandle(k)
      return computeLoss(J / J0, [volCons])

    # Optimization Loop
    for self.epoch in range(optParams['maxEpochs']):
      penal = min(8.0, 1. + self.epoch * 0.02)
      opt_state = opt_update(self.epoch,
                             optimizers.clip_grads(jax.grad(closure)(get_params(opt_state)), 1.),
                             opt_state)

      if self.epoch % 10 == 0:
        convgHistory['epoch'].append(self.epoch)
        mstrType, density = self.topNet.forward(get_params(opt_state), xyF)
        k = getThermalMatrix(mstrType, 0.01 + density)
        J = self.FE.objectiveHandle(k)
        convgHistory['J'].append(J)
        volf = jnp.mean(density)
        convgHistory['vf'].append(volf)
        if self.epoch == 10:
          J0 = J
        status = f'epoch \t {self.epoch} \t J \t {J:.2E} \t vf \t {volf:.2F}'
        print(status)
        if self.epoch % 30 == 0:
          self.FE.mesh.plotFieldOnMesh(density, status)

    if savedNet['isDump']:
      trained_params = optimizers.unpack_optimizer_state(opt_state)
      pickle.dump(trained_params, open(savedNet['file'], "wb"))

    return convgHistory

  #-----------------------#
  def plotCompositeTopology(self, res):
    """Plot composite topology with updated visualization for heat transfer optimization."""

    # Generate grid points
    xy = self.FE.mesh.generatePoints(res)
    xyS = applySymmetry(xy, self.symMap)
    xyE = applyExtrusion(xyS, self.extrusion)
    xyR = applyRotationalSymmetry(xyE, self.rotationalSymmetry)
    xyF = applyFourierMap(xyR, self.fourierMap) if self.fourierMap['isOn'] else xyR

    # Get microstructure types and density
    mstrType, density = self.topNet.forward(self.topNet.wts, xyF)

    # Define color maps for visualization
    fillColors = ['white', (0.8, 0, 0), (0, 0.8, 0), (0, 0, 0.8), (0, 0, 0), (0, 0.8, 0.8),
                  (0.8, 0, 0.8), (0.5, 0, 0.5), (1, 0.55, 0), (0, 0.5, 0.5), (0, 0, 0.5),
                  (0, 0.5, 0), (0.5, 0, 0), (0.5, 0.5, 0)]

    microstrImages = np.load('./microStrImages.npy')  # Load microstructure images

    # Define grid size
    NX = res * int(np.ceil((self.FE.mesh.bb['xmax'] - self.FE.mesh.bb['xmin']) / self.FE.mesh.elemSize[0]))
    NY = res * int(np.ceil((self.FE.mesh.bb['ymax'] - self.FE.mesh.bb['ymin']) / self.FE.mesh.elemSize[1]))
    nx, ny = microstrImages.shape[2], microstrImages.shape[3]

    # Initialize images
    compositeImg = np.zeros((NX * nx, NY * ny))
    conductivityImg = np.zeros((NX, NY))  # For thermal conductivity visualization
    temperatureImg = np.zeros((NX, NY))  # For temperature visualization
    maxC = 0
    step = 0.01  # Microstructure step
    cutOff = 0.98  # Threshold for fully dense regions

    # Compute heat transfer field (if available in the solver)
    temperature_field = self.FE.solveTemperatureField()

    for elem in range(xy.shape[0]):
      cx = int((res * xy[elem, 0]) / self.FE.mesh.elemSize[0])
      cy = int((res * xy[elem, 1]) / self.FE.mesh.elemSize[1])
      conductivityImg[cx, cy] = int(100. * density[elem])  # Use density as a proxy for conductivity

      # Store temperature field values
      temperatureImg[cx, cy] = temperature_field[elem] if temperature_field is not None else 0

      if density[elem] > cutOff:
        compositeImg[nx * cx:(cx + 1) * nx, ny * cy:(cy + 1) * ny] = np.ones((nx, ny))
      else:
        mstrIdx = min(microstrImages.shape[1] - 1, int(density[elem] // step))
        mstrTypeIdx = np.argmax(mstrType[elem, :])
        mstrimg = microstrImages[mstrTypeIdx, mstrIdx, :, :].T
        c = np.argmax(mstrType[elem, :]) + 1
        maxC = max(maxC, c)
        compositeImg[nx * cx:(cx + 1) * nx, ny * cy:(cy + 1) * ny] = mstrimg * c

    # Plot Composite Microstructure Topology
    plt.figure()
    plt.imshow(compositeImg.T, cmap=colors.ListedColormap(fillColors[:maxC + 1]),
               interpolation='none', vmin=0, vmax=maxC, origin='lower')
    plt.colorbar(label="Microstructure Types")
    plt.title("Optimized Composite Microstructure")
    plt.savefig(f'./top_{self.epoch:d}.pdf', dpi=300)
    plt.show()

    # Plot Thermal Conductivity Distribution
    plt.figure()
    plt.imshow(conductivityImg.T, cmap="coolwarm", interpolation='none', origin='lower')
    plt.colorbar(label="Thermal Conductivity (%)")
    plt.title("Thermal Conductivity Distribution")
    plt.show()

    # Plot Temperature Field
    plt.figure()
    plt.imshow(temperatureImg.T, cmap="hot", interpolation='none', origin='lower')
    plt.colorbar(label="Temperature (°C)")
    plt.title("Temperature Distribution")
    plt.show()
