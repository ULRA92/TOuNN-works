import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Set default font for plots
font = {'family': 'normal', 'size': 16}
matplotlib.rc('font', **font)

# Define microstructure thermal conductivity coefficients
microStrs = {
    'square': {'type': 'thermal', 'order': 5,
               'k': {'5': 0, '4': 1.5, '3': -2.2, '2': 1.1, '1': 0.4, '0': 0.0}},

    'xBox': {'type': 'thermal', 'order': 5,
             'k': {'5': 0, '4': 2.0, '3': -3.0, '2': 1.8, '1': 0.3, '0': 0.0}},

    'x': {'type': 'thermal', 'order': 5,
          'k': {'5': 0, '4': 1.8, '3': -2.5, '2': 1.3, '1': 0.5, '0': 0.0}},

    'diam': {'type': 'thermal', 'order': 5,
             'k': {'5': 0, '4': 1.3, '3': -1.8, '2': 0.9, '1': 0.6, '0': 0.0}},

    'star': {'type': 'thermal', 'order': 5,
             'k': {'5': 0, '4': 2.5, '3': -3.5, '2': 2.0, '1': 0.7, '0': 0.0}},

    'diamPlus': {'type': 'thermal', 'order': 5,
                 'k': {'5': 0, '4': 2.0, '3': -2.8, '2': 1.5, '1': 0.5, '0': 0.0}},

    'xDiam': {'type': 'thermal', 'order': 5,
              'k': {'5': 0, '4': 1.9, '3': -2.7, '2': 1.4, '1': 0.4, '0': 0.0}},

    'zBox': {'type': 'thermal', 'order': 5,
             'k': {'5': 0, '4': 2.2, '3': -3.2, '2': 1.6, '1': 0.6, '0': 0.0}},

    'nBox': {'type': 'thermal', 'order': 5,
             'k': {'5': 0, '4': 2.1, '3': -3.0, '2': 1.7, '1': 0.5, '0': 0.0}},

    'vDiam': {'type': 'thermal', 'order': 5,
              'k': {'5': 0, '4': 2.4, '3': -3.6, '2': 2.2, '1': 0.8, '0': 0.0}},

    'hDiam': {'type': 'thermal', 'order': 5,
              'k': {'5': 0, '4': 1.7, '3': -2.3, '2': 1.2, '1': 0.5, '0': 0.0}}
}


def plotInterpolateCoeffs():
    """
  Plot interpolated thermal conductivity as a function of volume fraction.
  """
    numPts = 100
    v = np.linspace(0, 1, numPts)

    # Thermal conductivity interpolation
    k_interp = {}
    for structure in microStrs:
        k_interp[structure] = np.zeros((numPts))
        for pw in range(5):
            k_interp[structure] += microStrs[structure]['k'][str(pw)] * (v ** pw)

    # Plot results
    plt.figure()
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'black', 'pink']
    markers = ['o', 's', 'd', 'h', '*', 'v', '^']
    linestyles = [':', '--', '-', '-.', '-', '--', ':']

    for ctr, structure in enumerate(k_interp):
        plt.plot(v, k_interp[structure], color=colors[ctr % len(colors)],
                 linestyle=linestyles[ctr % len(linestyles)],
                 marker=markers[ctr % len(markers)],
                 markevery=20, label=f'{structure}')

    plt.xlabel('Volume Fraction (v)')
    plt.ylabel('Thermal Conductivity (k)')
    plt.legend()
    plt.title("Interpolated Thermal Conductivity for Microstructures")
    plt.show()


plotInterpolateCoeffs()
