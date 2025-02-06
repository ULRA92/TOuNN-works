import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Helvetica','Times New Roman'
import numpy as np


def plotConvergence(convg):
    """
  Plots convergence metrics for heat transfer optimization.

  - Uses semilog scale for thermal resistance minimization.
  - Ensures temperature field plots remain in linear scale.
  """
    x = np.array(convg['epoch'])

    for key in convg:
        if key == 'epoch':
            continue  # 'epoch' is the x-axis for all plots

        plt.figure(figsize=(7, 5))
        y = np.array(convg[key])

        # Use logarithmic scale for thermal resistance, linear for temperature
        if 'resistance' in key.lower():
            plt.semilogy(x, y, label=str(key), color='b', linestyle='--')
            plt.ylabel(f"{key} (log scale)")
        else:
            plt.plot(x, y, label=str(key), color='r', linestyle='-')
            plt.ylabel(str(key))

        plt.xlabel('Iterations')
        plt.title(f"Convergence of {key}")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()


def plotTemperatureField(temperature, mesh):
    """
  Plots the final temperature distribution.

  - Uses a heatmap for temperature field visualization.
  """
    if temperature is None or mesh is None:
        print("Temperature field or mesh data not provided.")
        return

    plt.figure(figsize=(8, 6))
    plt.tricontourf(mesh[:, 0], mesh[:, 1], temperature, levels=50, cmap='hot')
    plt.colorbar(label="Temperature (°C)")
    plt.xlabel("X-axis (m)")
    plt.ylabel("Y-axis (m)")
    plt.title("Final Temperature Distribution")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
