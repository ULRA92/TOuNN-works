import numpy as np
import jax.numpy as jnp
import jax
from jax import jit
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import colors
from FE_Solver import JAXSolver
from network import TopNet
from projections import applyFourierMap, applySymmetry, applyRotationalSymmetry, applyExtrusion
from jax.example_libraries import optimizers
from materialCoeffs import microStrs, getThermalConductivity
import pickle


class TOuNN:
    def __init__(self, exampleName, mesh, material, nnSettings, symMap, fourierMap, rotationalSymmetry, extrusion):
        self.exampleName = exampleName
        self.FE = JAXSolver(mesh, material)
        self.xy = self.FE.mesh.elemCenters
        self.fourierMap = fourierMap
        self.topNet = TopNet(nnSettings)
        self.symMap = symMap
        self.mstrData = microStrs
        self.rotationalSymmetry = rotationalSymmetry
        self.extrusion = extrusion

        # ✅ Define JIT function inside __init__ to access `self.FE`
        @jit
        def jit_objective(getThermalMatrix, mstrType, nn_rho):
            return self.FE.objective(getThermalMatrix(mstrType, nn_rho))

        self.objectiveHandle = partial(jit_objective)  # ✅ Now it's inside __init__


    def getThermalMatrix(self, mstrType, nn_rho):
        vfracPow = {str(pw): nn_rho ** pw for pw in range(self.mstrData.get('square', {}).get('order', 0) + 1)}
        k = jnp.zeros((self.FE.mesh.numElems))

        for mstrCtr, mstr in enumerate(self.mstrData):
            k_mstr = getThermalConductivity(vfracPow, self.mstrData[mstr])
            k = k.at[:].add(jnp.einsum('i,i->i', mstrType[:, mstrCtr], k_mstr))

        return k

    def optimizeDesign(self, optParams, savedNet):
        convgHistory = {'epoch': [], 'vf': [], 'J': []}
        opt_init, opt_update, get_params = optimizers.adam(optParams['learningRate'])
        opt_state = opt_init(self.topNet.wts)
        opt_update = jax.jit(opt_update, static_argnums=(0,))  # Prevents re-compilation every step

        if savedNet['isAvailable']:
            saved_params = pickle.load(open(savedNet['file'], "rb"))
            opt_state = optimizers.pack_optimizer_state(saved_params)

        xyS = applySymmetry(self.xy, self.symMap)
        xyE = applyExtrusion(xyS, self.extrusion)
        xyR = applyRotationalSymmetry(xyE, self.rotationalSymmetry)
        xyF = applyFourierMap(xyR, self.fourierMap) if self.fourierMap['isOn'] else xyR

        mstrType, density0 = self.topNet.forward(get_params(opt_state), xyF)
        J0 = self.objectiveHandle(self.getThermalMatrix, mstrType, 0.01 + density0)
        print(f"DEBUG: xyF type = {type(xyF)}, shape = {xyF.shape}")
        print(f"DEBUG: First 5 rows of xyF = {xyF[:5]}")

        def computeLoss(objective, constraints):
            loss = objective
            if optParams['lossMethod']['type'] == 'penalty':
                alpha = optParams['lossMethod']['alpha0'] + self.epoch * optParams['lossMethod']['delAlpha']
                for c in constraints:
                    loss += alpha * c ** 2
            elif optParams['lossMethod']['type'] == 'logBarrier':
                t = optParams['lossMethod']['t0'] * optParams['lossMethod']['mu'] ** self.epoch
                for c in constraints:
                    loss += (-jnp.log(-c) / t) if c < (-1 / t ** 2) else (t * c - jnp.log(1 / t ** 2) / t + 1 / t)
            return loss

        def closure(nnwts, J0=J0):
            mstrType, density = self.topNet.forward(nnwts, xyF)
            volCons = (jnp.mean(density) / optParams['desiredVolumeFraction']) - 1.
            k = jnp.array(self.getThermalMatrix(mstrType, 0.01 + density))
            J = self.FE.objective(k)
            return computeLoss(J / J0, [volCons])

        for self.epoch in range(optParams['maxEpochs']):
            opt_state = opt_update(self.epoch, jax.grad(closure)(get_params(opt_state)), opt_state)

            if self.epoch % 50 == 0:
                convgHistory['epoch'].append(self.epoch)
                mstrType, density = self.topNet.forward(get_params(opt_state), xyF)
                k = jnp.array(self.getThermalMatrix(mstrType, 0.01 + density))
                J = self.FE.objective(k)
                convgHistory['J'].append(J)
                convgHistory['vf'].append(jnp.mean(density))
                print(f'epoch {self.epoch} \t J {J:.2E} \t vf {jnp.mean(density):.2F}')
                tempField = self.FE.solveTemperatureField()
                self.FE.mesh.plotFieldOnMesh(tempField, titleStr=f"Temperature Field at Epoch {self.epoch}")
                plt.pause(0.1)

        if savedNet['isDump']:
            trained_params = optimizers.unpack_optimizer_state(opt_state)
            pickle.dump(trained_params, open(savedNet['file'], "wb"))

        return convgHistory


#-----------------------#
def plotCompositeTopology(self, res):
    """Plot composite topology with updated visualization for heat transfer optimization."""
    print("DEBUG: Running plotCompositeTopology function")

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

    print("DEBUG: Loading microstructure images")
    microstrImages = np.load('./microStrImages.npy')  # Load microstructure images

    # Define grid size
    NX = int(np.ceil((self.FE.mesh.bb['xmax'] - self.FE.mesh.bb['xmin']) / self.FE.mesh.elemSize[0]))
    NY = int(np.ceil((self.FE.mesh.bb['ymax'] - self.FE.mesh.bb['ymin']) / self.FE.mesh.elemSize[1]))
    nx, ny = microstrImages.shape[2], microstrImages.shape[3]

    print(f"DEBUG: Grid size (NX, NY) = ({NX}, {NY})")

    # Initialize images
    compositeImg = np.zeros((NX * nx, NY * ny))
    conductivityImg = np.zeros((NX, NY))  # For thermal conductivity visualization
    temperatureImg = np.zeros((NX, NY))  # For temperature visualization
    maxC = 0
    step = 0.01  # Microstructure step
    cutOff = 0.98  # Threshold for fully dense regions

    # Compute heat transfer field (if available in the solver)
    print("DEBUG: Solving temperature field")
    temperature_field = self.FE.solveTemperatureField()

    for elem in range(xy.shape[0]):
      cx = min(NX - 1, int((xy[elem, 0] - self.FE.mesh.bb['xmin']) / self.FE.mesh.elemSize[0]))
      cy = min(NY - 1, int((xy[elem, 1] - self.FE.mesh.bb['ymin']) / self.FE.mesh.elemSize[1]))
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

    print("DEBUG: Plotting Composite Microstructure Topology")
    plt.figure(figsize=(8, 6))
    plt.imshow(compositeImg.T, cmap=colors.ListedColormap(fillColors[:maxC + 1]),
               interpolation='none', vmin=0, vmax=maxC, origin='lower')
    plt.colorbar(label="Microstructure Types")
    plt.title("Optimized Composite Microstructure", fontsize=14)
    plt.xlabel("X-axis", fontsize=12)
    plt.ylabel("Y-axis", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'./top_{self.epoch:d}.pdf', dpi=300)
    plt.close()

    print("DEBUG: Plotting Thermal Conductivity Distribution")
    plt.figure(figsize=(8, 6))
    plt.imshow(conductivityImg.T, cmap="coolwarm", interpolation='none', origin='lower')
    plt.colorbar(label="Thermal Conductivity (%)")
    plt.title("Thermal Conductivity Distribution", fontsize=14)
    plt.xlabel("X-axis", fontsize=12)
    plt.ylabel("Y-axis", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    plt.close()

    print("DEBUG: Plotting Temperature Field")
    plt.figure(figsize=(8, 6))
    plt.imshow(temperatureImg.T, cmap="hot", interpolation='none', origin='lower')
    plt.colorbar(label="Temperature (°C)")
    plt.title("Temperature Distribution", fontsize=14)
    plt.xlabel("X-axis", fontsize=12)
    plt.ylabel("Y-axis", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    plt.close()

    print("DEBUG: Plotting completed successfully")

    # ✅ **Keep the Final Plots Open for Review**
    plt.show(block=True)  # Ensure final plots remain visible

