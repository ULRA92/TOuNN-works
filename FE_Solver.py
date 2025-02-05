import jax.numpy as jnp
import numpy as np
from jax import jit
import jax
from FE_templates import getKMatrixGridMeshTemplates

class JAXSolver:
    def __init__(self, mesh, material):
        self.mesh = mesh
        self.material = material
        self.Ktemplates = getKMatrixGridMeshTemplates(mesh.elemSize, 'thermal')  # Updated for heat transfer
        self.objectiveHandle = jit(self.objective)
        self.k0 = self.material.getThermalConductivityMatrix(self.mesh)  # Use thermal conductivity instead of stiffness

    # -----------------------#
    def objective(self, k_field):
        """
        Objective function for heat transfer optimization.
        - Minimizes thermal resistance by solving the heat conduction equation.
        - k_field: Element-wise thermal conductivity distribution.
        """

        @jit
        def assembleK(k_field):
            """Assemble the global heat transfer matrix K (conductivity matrix)."""
            sK = jnp.zeros((self.mesh.numElems, 4, 4))  # 4 DOFs per element in heat transfer problems
            for k in k_field:
                sK += jnp.einsum('e,jk->ejk', k_field[k], self.Ktemplates[k])  # Modify for heat transfer

            K = jnp.zeros((self.mesh.ndof, self.mesh.ndof))
            K = K.at[self.mesh.nodeIdx].add(sK.flatten())  # Updated for JAX compatibility
            return K

        # -----------------------#
        @jit
        def solve(K):
            """
            Solves for temperature field using:
            K * T = Q (Fourier's heat conduction equation).
            - K: Global heat transfer matrix.
            - T: Temperature field.
            - Q: Heat source vector.
            """
            T_free = jax.scipy.linalg.solve(
                K[self.mesh.bc['free'], :][:, self.mesh.bc['free']],
                self.mesh.bc['heat'][self.mesh.bc['free']],  # Heat source instead of force
                assume_a='pos', check_finite=False
            )
            T = jnp.zeros((self.mesh.ndof))
            T = T.at[self.mesh.bc['free']].set(T_free.reshape(-1))  # Updated
            return T

        # -----------------------#
        @jit
        def computeThermalObjective(K, T):
            """
            Compute the objective function for heat transfer:
            - Default: Minimize total thermal resistance (gradient of temperature).
            - Alternative: Minimize max temperature (to avoid hotspots).
            """
            thermal_resistance = jnp.sum(K * jnp.square(jnp.gradient(T)))
            return thermal_resistance

        # -----------------------#
        K = assembleK(k_field)  # Assemble global heat transfer matrix
        T = solve(K)  # Solve for temperature field
        J = computeThermalObjective(K, T)  # Compute objective (thermal resistance)
        return J

    # -----------------------#
    def solveTemperatureField(self):
        """
        Solves the temperature distribution based on the given material distribution.
        Returns: Nodal temperature field.
        """
        @jit
        def assembleK():
            """Assemble the heat transfer matrix."""
            sK = jnp.zeros((self.mesh.numElems, 4, 4))  # 4 DOFs per element for temperature
            for k in self.k0:
                sK += jnp.einsum('e,jk->ejk', self.k0[k], self.Ktemplates[k])

            K = jnp.zeros((self.mesh.ndof, self.mesh.ndof))
            K = K.at[self.mesh.nodeIdx].add(sK.flatten())
            return K

        @jit
        def solve(K):
            """Solve the temperature field T from K*T = Q."""

            T_free = jax.scipy.linalg.solve(
                K[self.mesh.bc['free'], :][:, self.mesh.bc['free']],
                self.mesh.bc['heat'][self.mesh.bc['free']],
                assume_a='pos', check_finite=False
            )
            T = jnp.zeros((self.mesh.ndof))
            T = T.at[self.mesh.bc['free']].set(T_free.reshape(-1))
            return T

        K = assembleK()  # Assemble global conductivity matrix
        T = solve(K)  # Solve for temperature field
        return T  # Return temperature distribution

    # -----------------------#
