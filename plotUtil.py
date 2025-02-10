import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import numpy as np

def plotConvergence(convg):
    x = np.array(convg['epoch'])

    for key in convg:
        if key == 'epoch':
            continue

        plt.figure(figsize=(10, 6))
        y = np.array(convg[key])

        if 'resistance' in key.lower():
            plt.semilogy(x, y, label=str(key), color='b', linestyle='--')
        else:
            plt.plot(x, y, label=str(key), linestyle='-', marker='o', markersize=6)

        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel(str(key), fontsize=14)
        plt.title(f"Convergence of {key}", fontsize=16)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12, frameon=True)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

def plotTemperatureField(temperature, mesh):
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
