import numpy as np

class Material:
    def __init__(self, matProp):
        """
        Initializes material properties.

        - matProp: Dictionary containing material properties (Emax, nu, k_max, k_min).
        """
        self.matProp = matProp

        # Elastic modulus properties (for structural cases)
        E, nu = matProp.get('Emax', 1), matProp.get('nu', 0.3)
        self.C = E / (1 - nu ** 2) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])

        # Thermal properties
        self.k_max = matProp.get('k_max', 10)  # Maximum thermal conductivity
        self.k_min = matProp.get('k_min', 0.1)  # Minimum thermal conductivity

    # --------------------------#
    def computeSIMP_Interpolation(self, rho, penal):
        """
        SIMP interpolation for structural Young's modulus.
        """
        E = 0.001 * self.matProp['Emax'] + (0.999 * self.matProp['Emax']) * (rho + 0.01) ** penal
        return E

    # --------------------------#
    def computeRAMP_Interpolation(self, rho, penal):
        """
        RAMP interpolation for structural Young's modulus.
        """
        E = 0.001 * self.matProp['Emax'] + (0.999 * self.matProp['Emax']) * (rho / (1. + penal * (1. - rho)))
        return E

    # --------------------------#
    def computeThermalSIMP_Interpolation(self, rho, penal):
        """
        SIMP interpolation for thermal conductivity.
        """
        k = self.k_min + (self.k_max - self.k_min) * (rho + 0.01) ** penal
        return k

    # --------------------------#
    def computeThermalRAMP_Interpolation(self, rho, penal):
        """
        RAMP interpolation for thermal conductivity.
        """
        k = self.k_min + (self.k_max - self.k_min) * (rho / (1. + penal * (1. - rho)))
        return k

    # --------------------------#
    def getThermalConductivityMatrix(self, mesh):
        """
        Returns base element-wise thermal conductivity matrix for grid-based meshes.
        """
        if mesh.meshType == 'gridMesh':
            k_base = 1  # Base conductivity value (normalized)
            k_template = np.array([
                [1 / 3, -1 / 3, -1 / 6, 1 / 6],
                [-1 / 3, 1 / 3, 1 / 6, -1 / 6],
                [-1 / 6, 1 / 6, 1 / 3, -1 / 3],
                [1 / 6, -1 / 6, -1 / 3, 1 / 3]
            ])
            return k_base * k_template  # Uniform conductivity

        return None  # Default for unsupported mesh types
