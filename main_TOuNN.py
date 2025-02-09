import os
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import configparser

from examples import getExampleBC
from Mesher import RectangularGridMesher
from projections import computeFourierMap
from material import Material
from TOuNN import TOuNN
from plotUtil import plotConvergence, plotTemperatureField

# ==================== 🔹 JAX MEMORY FIXES ====================
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# ==================== 🔹 READ CONFIGURATION ====================
configFile = './config.txt'
config = configparser.ConfigParser()
config.read(configFile)

# ==================== 🔹 MESH AND BOUNDARY CONDITIONS ====================
meshConfig = config['MESH']
ndim = meshConfig.getint('ndim')
nelx = meshConfig.getint('nelx')
nely = meshConfig.getint('nely')
elemSize = np.array(meshConfig['elemSize'].split(',')).astype(float)

exampleName, bcSettings, symMap = getExampleBC(1, nelx, nely, elemSize)
mesh = RectangularGridMesher(ndim, nelx, nely, elemSize, bcSettings, physics='thermal')

mesh.processBoundaryCondition(
    fixedTempNodes=bcSettings.get('fixedTemperatureNodes', []),
    heatFluxNodes=bcSettings.get('heatFluxNodes', []),
    heatSourceNodes=bcSettings.get('heatSourceNodes', [])
)

# ==================== 🔹 MATERIAL PROPERTIES ====================
materialConfig = config['MATERIAL']
k_max, k_min = materialConfig.getfloat('k_max'), materialConfig.getfloat('k_min')
matProp = {'physics': 'thermal', 'k_max': k_max, 'k_min': k_min}
material = Material(matProp)

# ==================== 🔹 NEURAL NETWORK SETTINGS ====================
tounnConfig = config['TOUNN']
nnSettings = {
    'numLayers': tounnConfig.getint('numLayers'),
    'numNeuronsPerLayer': tounnConfig.getint('hiddenDim'),
    'outputDim': tounnConfig.getint('outputDim')
}

fourierMap = {
    'isOn': tounnConfig.getboolean('fourier_isOn'),
    'minRadius': tounnConfig.getfloat('fourier_minRadius'),
    'maxRadius': tounnConfig.getfloat('fourier_maxRadius'),
    'numTerms': tounnConfig.getint('fourier_numTerms')
}
fourierMap['map'] = computeFourierMap(mesh, fourierMap)

# ==================== 🔹 OPTIMIZATION PARAMETERS ====================
lossConfig = config['LOSS']
lossMethod = {
    'type': 'penalty',
    'alpha0': lossConfig.getfloat('alpha0'),
    'delAlpha': lossConfig.getfloat('delAlpha')
}

optConfig = config['OPTIMIZATION']
optParams = {
    'maxEpochs': optConfig.getint('numEpochs'),
    'lossMethod': lossMethod,
    'learningRate': optConfig.getfloat('lr'),
    'desiredVolumeFraction': optConfig.getfloat('desiredVolumeFraction'),
    'gradclip': {
        'isOn': optConfig.getboolean('gradClip_isOn'),
        'thresh': optConfig.getfloat('gradClip_clipNorm')
    }
}

rotationalSymmetry = {'isOn': False, 'sectorAngleDeg': 90, 'centerCoordn': np.array([20, 10])}
extrusion = {'X': {'isOn': False, 'delta': 1.}, 'Y': {'isOn': False, 'delta': 1.}}

dummy_k = jnp.ones((mesh.numElems,), dtype=jnp.float32)
dummy_tounn = TOuNN(exampleName, mesh, material, nnSettings, symMap, fourierMap, rotationalSymmetry, extrusion)

# **🚀 Precompile JIT before main execution**
try:
    dummy_tounn.FE.objectiveHandle(dummy_k)
    print("DEBUG: JIT Compilation Successful")
except Exception as e:
    print(f"ERROR: JIT Compilation Failed - {e}")

plt.close('all')
savedNet = {'isAvailable': False, 'file': './netWeights.pkl', 'isDump': False}

start = time.perf_counter()
tounn = TOuNN(exampleName, mesh, material, nnSettings, symMap, fourierMap, rotationalSymmetry, extrusion)

# 🏆 Optimization Loop with Intermediate Plotting
convgHistory = []
for epoch in range(optParams['maxEpochs']):
    loss = tounn.optimizeDesign(optParams, savedNet)
    convgHistory.append(loss)

    if epoch % 50 == 0:
        tempField = tounn.FE.solveTemperatureField()
        mesh.plotFieldOnMesh(tempField, titleStr=f"Temperature Field at Iteration {epoch}")

print(f'Time taken (sec): {time.perf_counter() - start:.2F}')

# ==================== 🔹 PLOT CONVERGENCE HISTORY ====================
plotConvergence(convgHistory)

# ==================== 🔹 COMPUTE FINAL TEMPERATURE FIELD ====================
finalTemperature = tounn.FE.solveTemperatureField()

# ==================== 🔹 PLOT FINAL TEMPERATURE DISTRIBUTION ====================
plotTemperatureField(finalTemperature, mesh.nodeXY)

plt.show(block=True)
