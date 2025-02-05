import numpy as np

#  ~~~~~~~~~~~~ Heat Transfer Examples ~~~~~~~~~~~~~#
def getExampleBC(example, nelx, nely, elemSize):
    """
    Define boundary conditions and symmetry maps for heat transfer examples.

    - Heat flux replaces force boundary conditions.
    - Fixed temperature replaces displacement constraints.
    """
    if example == 1:  # Heat Conduction in a Rod (Tip Cantilever Equivalent)
        exampleName = 'HeatRod'
        bcSettings = {
            'fixedTemperatureNodes': np.arange(0, (nely+1), 1),  # Fix temperature at one end
            'heatFluxNodes': [(nelx+1)*(nely+1)-1],  # Apply heat flux at the other end
            'heatMagnitude': 100.,  # Heat flux magnitude
            'dofsPerNode': 1
        }
        symMap = {'XAxis': {'isOn': False}, 'YAxis': {'isOn': False}}

    elif example == 2:  # Heated Plate with Fixed Edges (Mid Cantilever Equivalent)
        exampleName = 'HeatedPlate'
        bcSettings = {
            'fixedTemperatureNodes': np.arange(0, (nely+1), 1),  # Fix temperature at bottom edge
            'heatFluxNodes': np.arange((nelx+1)*(nely+1)- (nely+1), (nelx+1)*(nely+1)),  # Apply heat flux at top edge
            'heatMagnitude': 50.,
            'dofsPerNode': 1
        }
        symMap = {'XAxis': {'isOn': True}, 'YAxis': {'isOn': False}}

    elif example == 3:  # Heat Sink Model (MBB Beam Equivalent)
        exampleName = 'HeatSink'
        fixedNodes = np.union1d(np.arange(0, (nely+1), 1), (nelx+1)*(nely+1)- (nely+1))
        bcSettings = {
            'fixedTemperatureNodes': fixedNodes,  # Fix temperature at two sides
            'heatFluxNodes': [2*(nely+1)+1],  # Apply heat flux in the middle
            'heatMagnitude': 200.,
            'dofsPerNode': 1
        }
        symMap = {'XAxis': {'isOn': False}, 'YAxis': {'isOn': False}}

    elif example == 4:  # Heat Diffusion in a Cross-section (Michell Equivalent)
        exampleName = 'HeatCrossSection'
        fixedNodes = np.array([0, 1, (nelx+1)*(nely+1)-nely])
        bcSettings = {
            'fixedTemperatureNodes': fixedNodes,
            'heatFluxNodes': [nelx*(nely+1)+1],
            'heatMagnitude': 150.,
            'dofsPerNode': 1
        }
        symMap = {'XAxis': {'isOn': False}, 'YAxis': {'isOn': True}}

    elif example == 5:  # Distributed Heat Transfer in a Bridge-like Structure
        exampleName = 'HeatBridge'
        fixedNodes = np.array([0, 1, (nelx+1)*(nely+1)-nely, (nelx+1)*(nely+1)-nely+1])
        heatFluxNodes = np.arange(2*nely+1, (nelx+1)*(nely+1), 8*(nely+1))
        bcSettings = {
            'fixedTemperatureNodes': fixedNodes,
            'heatFluxNodes': heatFluxNodes,
            'heatMagnitude': 10.,  # Distributed heat input
            'dofsPerNode': 1
        }
        symMap = {'XAxis': {'isOn': False}, 'YAxis': {'isOn': True}}

    elif example == 6:  # Heat Distribution in a Tensile Bar
        exampleName = 'HeatedBar'
        fixedNodes = np.union1d(np.arange(0, (nely+1), 1), [1])
        midDofX = (nelx+1)*(nely+1)- (nely)
        heatFluxNodes = np.arange(midDofX-6, midDofX+6, 2)
        bcSettings = {
            'fixedTemperatureNodes': fixedNodes,
            'heatFluxNodes': heatFluxNodes,
            'heatMagnitude': 75.,
            'dofsPerNode': 1
        }
        symMap = {'XAxis': {'isOn': True}, 'YAxis': {'isOn': False}}

    elif example == 7:  # Large Heat Distribution Over a Surface
        exampleName = 'FullHeatSurface'
        heatFluxNodes = np.arange((nelx+1)*(nely+1)-nely+1, (nelx+1)*(nely+1), 2)
        bcSettings = {
            'fixedTemperatureNodes': np.arange(0, (nely+1), 1),
            'heatFluxNodes': heatFluxNodes,
            'heatMagnitude': 500.,
            'dofsPerNode': 1
        }
        symMap = {'XAxis': {'isOn': True}, 'YAxis': {'isOn': False}}

    elif example == 8:  # Torsional Heating
        exampleName = 'TorsionalHeating'
        bcSettings = {
            'fixedTemperatureNodes': np.loadtxt('./Mesh/Torsion/fixed.txt').astype(int),
            'heatFluxNodes': np.loadtxt('./Mesh/Torsion/force.txt').astype(int),
            'heatMagnitude': 300.,
            'dofsPerNode': 1
        }
        symMap = {'XAxis': {'isOn': True}, 'YAxis': {'isOn': True}}

    elif example == 9:  # L-Shaped Heated Structure
        exampleName = 'LHeatBracket'
        bcSettings = {
            'fixedTemperatureNodes': np.loadtxt('./Mesh/LBracket/fixed.txt').astype(int),
            'heatFluxNodes': np.loadtxt('./Mesh/LBracket/force.txt').astype(int),
            'heatMagnitude': 250.,
            'dofsPerNode': 1
        }
        symMap = {'XAxis': {'isOn': False}, 'YAxis': {'isOn': False}}

    elif example == 10:  # Mid-Loaded Heat Transfer in MBB Beam
        exampleName = 'MidLoadHeatMBB'
        bcSettings = {
            'fixedTemperatureNodes': np.loadtxt('./Mesh/midLoadMBB/fixed.txt').astype(int),
            'heatFluxNodes': np.loadtxt('./Mesh/midLoadMBB/force.txt').astype(int),
            'heatMagnitude': 350.,
            'dofsPerNode': 1
        }
        symMap = {'XAxis': {'isOn': False}, 'YAxis': {'isOn': True}}

    return exampleName, bcSettings, symMap
