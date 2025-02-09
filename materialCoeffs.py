import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.font_manager as fm
# Set default font for plots
font = {'family': 'normal', 'size': 16}
matplotlib.rc('font', **font)

# Pick a font that exists
available_fonts = sorted(set(f.name for f in fm.fontManager.ttflist))
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


def getThermalConductivity(vfracPow, microstructure):
    """
    Computes the interpolated thermal conductivity based on microstructure and volume fraction.

    - `vfracPow`: Dictionary containing precomputed powers of volume fraction (`v^p`).
    - `microstructure`: Microstructure dictionary from `microStrs`.

    Returns:
        Interpolated thermal conductivity.
    """
    k_interp = jnp.zeros_like(list(vfracPow.values())[0])  # Initialize conductivity array

    for pw in range(microstructure['order'] + 1):
        k_interp += microstructure['k'][str(pw)] * vfracPow[str(pw)]

    return k_interp


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
                 markevery=5, label=f'{structure}')

    plt.xlabel('Volume Fraction (v)')
    plt.ylabel('Thermal Conductivity (k)')
    plt.legend()
    plt.title("Interpolated Thermal Conductivity for Microstructures")
    plt.show()


# Prevent `plotInterpolateCoeffs()` from running unless this file is executed directly
if __name__ == "__main__":
    plotInterpolateCoeffs()