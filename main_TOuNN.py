import os
import jax.numpy as jnp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time
import configparser

from examples import getExampleBC
from Mesher import RectangularGridMesher, UnstructuredMesher
from projections import computeFourierMap
from material import Material
from TOuNN import TOuNN
from plotUtil import plotConvergence, plotTemperatureField

# ==================== 🔹 JAX MEMORY FIXES ====================
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"  # Use only 30% of available memory
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# ==================== 🔹 MATPLOTLIB FONT FIXES ====================
# Get all available fonts
"""
available_fonts = sorted(set(f.name for f in fm.fontManager.ttflist))

# Pick a font that exists
chosen_font = None
for preferred_font in ["Arial", "DejaVu Sans", "Calibri", "Verdana", "Times New Roman"]:
    if preferred_font in available_fonts:
        chosen_font = preferred_font
        break

if chosen_font is None:
    print("WARNING: No preferred font found. Using DejaVu Sans.")
    chosen_font = "DejaVu Sans"  # Ensure a working fallback

# 🔹 Force Matplotlib to use the selected font
matplotlib.rcParams.update({
    "font.family": chosen_font,
    "axes.unicode_minus": False,  # Fixes minus sign issue in plots
})
"""
#print(f"DEBUG: Using font: {chosen_font}")

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

# Use structured or unstructured meshing
mesh = RectangularGridMesher(ndim, nelx, nely, elemSize, bcSettings, physics='thermal')

# Apply thermal boundary conditions
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
    'outputDim': tounnConfig.getint('outputDim'),
    'inputDim': 2

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

# Projection Settings
rotationalSymmetry = {'isOn': False, 'sectorAngleDeg': 90, 'centerCoordn': np.array([20, 10])}
extrusion = {'X': {'isOn': False, 'delta': 1.}, 'Y': {'isOn': False, 'delta': 1.}}

# ==================== 🔹 RUN OPTIMIZATION ====================
dummy_k = jnp.ones((mesh.numElems,), dtype=jnp.float32)  # Use float32 instead of default float64
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
convgHistory = []  # Initialize empty convergence history list

for epoch in range(optParams['maxEpochs']):
    loss = tounn.optimizeDesign(optParams, savedNet) # Run one step and capture loss
    convgHistory.append(loss)  # Store loss history for plotting

    if epoch % 50 == 0:  # Plot every 50 iterations

        tempField = tounn.FE.solveTemperatureField()
        mesh.plotFieldOnMesh(tempField, titleStr=f"Temperature Field at Iteration {epoch}")

print(f'Time taken (sec): {time.perf_counter() - start:.2F}')

# ==================== 🔹 PLOT CONVERGENCE HISTORY ====================
plotConvergence(convgHistory)  # Now properly defined

# ==================== 🔹 COMPUTE FINAL TEMPERATURE FIELD ====================
finalTemperature = tounn.FE.solveTemperatureField()

# ==================== 🔹 PLOT FINAL TEMPERATURE DISTRIBUTION ====================
plotTemperatureField(finalTemperature, mesh.nodeXY)

plt.show(block=True)  # Keep the final plot open
