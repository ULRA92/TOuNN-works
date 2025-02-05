import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Times New Roman' # Or 'Helvetica' / 'Times New Roman'
from examples import getExampleBC
from Mesher import RectangularGridMesher, UnstructuredMesher
from projections import computeFourierMap
from material import Material
from TOuNN import TOuNN
from plotUtil import plotConvergence, plotTemperatureField
import time
import configparser

# %% Read Configuration File
configFile = './config.txt'
config = configparser.ConfigParser()
config.read(configFile)

# %% Mesh and BC (Heat Transfer Case)
meshConfig = config['MESH']
ndim = meshConfig.getint('ndim')  # Default: 2D
nelx = meshConfig.getint('nelx')  # Number of elements along X
nely = meshConfig.getint('nely')  # Number of elements along Y
elemSize = np.array(meshConfig['elemSize'].split(',')).astype(float)

exampleName, bcSettings, symMap = getExampleBC(1, nelx, nely, elemSize)

# Select structured or unstructured meshing
mesh = RectangularGridMesher(ndim, nelx, nely, elemSize, bcSettings, physics='thermal')
# mesh = UnstructuredMesher(bcSettings, physics='thermal')

# Apply thermal boundary conditions
mesh.processBoundaryCondition(
    fixedTempNodes=bcSettings.get('fixedTemperatureNodes', []),
    heatFluxNodes=bcSettings.get('heatFluxNodes', []),
    heatSourceNodes=bcSettings.get('heatSourceNodes', [])
)

# %% Material (Thermal Conductivity)
materialConfig = config['MATERIAL']
k_max, k_min = materialConfig.getfloat('k_max'), materialConfig.getfloat('k_min')
matProp = {'physics': 'thermal', 'k_max': k_max, 'k_min': k_min}
material = Material(matProp)

# %% Neural Network Settings
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

# %% Optimization Parameters
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

# Other Projection Settings
rotationalSymmetry = {'isOn': False, 'sectorAngleDeg': 90, 'centerCoordn': np.array([20, 10])}
extrusion = {'X': {'isOn': False, 'delta': 1.}, 'Y': {'isOn': False, 'delta': 1.}}

# %% Run Optimization
plt.close('all')
savedNet = {'isAvailable': False, 'file': './netWeights.pkl', 'isDump': False}

start = time.perf_counter()
tounn = TOuNN(exampleName, mesh, material, nnSettings, symMap, fourierMap, rotationalSymmetry, extrusion)
convgHistory = tounn.optimizeDesign(optParams, savedNet)

print(f'Time taken (sec): {time.perf_counter() - start:.2F}')

# Plot Convergence History (Thermal Optimization)
plotConvergence(convgHistory)

# Compute Final Temperature Field
finalTemperature = tounn.FE.solveTemperatureField()

# Plot Final Temperature Distribution
plotTemperatureField(finalTemperature, mesh.nodeXY)

plt.show(block=True)
