import jax.numpy as jnp
import jax
from jax import jit
import jax.scipy.linalg
from FE_templates import getKMatrixGridMeshTemplates  # Ensure this supports 'thermal' physics

class JAXSolver:
    def __init__(self, mesh, material):
        self.mesh = mesh
        self.material = material
        self.Ktemplates = getKMatrixGridMeshTemplates(mesh.elemSize, 'thermal')  # Updated for heat transfer
        self.objectiveHandle = jax.jit(self.objective)
        self.k0 = self.material.getThermalConductivityMatrix(self.mesh)  # Use thermal conductivity instead of stiffness

    # -----------------------#
    def computeElementMatrix(self, k):
        """
        Compute the **heat transfer element stiffness matrix** for a given thermal conductivity `k`.
        - Uses predefined `Ktemplates`.
        """
        element_matrix = jnp.zeros((4, 4))  # 4 DOFs per element in heat transfer problems
        for key in self.Ktemplates:
            element_matrix += k * self.Ktemplates[key]
        return element_matrix

    # -----------------------#
    def assembleK(self, k_field):
        """Assemble the global heat transfer matrix K (conductivity matrix)."""
        sK = jnp.zeros((self.mesh.numElems, 4, 4))  # 4 DOFs per element in heat transfer problems
        k_field = jnp.asarray(k_field)  # ✅ Ensure it's a JAX array

        for elem in range(self.mesh.numElems):
            k = k_field[elem]
            element_matrix = self.computeElementMatrix(k)  # Get element matrix
            sK = sK.at[elem].set(element_matrix)

        # Build global matrix K
        K = jnp.zeros((self.mesh.ndof, self.mesh.ndof))
        iK, jK = map(jnp.array, self.mesh.nodeIdx)  # ✅ Fix indexing issue

        for elem in range(self.mesh.numElems):
            for i in range(4):
                for j in range(4):
                    K = K.at[iK[elem * 4 + i], jK[elem * 4 + j]].add(sK[elem, i, j])  # 🔥 Fixed indexing!

        return K

    # -----------------------#
    def objective(self, k_field):
        """Objective function for heat transfer optimization."""
        k_field = jnp.asarray(k_field)  # ✅ Ensure it’s a JAX array
        K = self.assembleK(k_field)
        T = self.solve(K)  # Ensure solve() is correctly referenced
        J = self.computeThermalObjective(K, T)
        return J

    # -----------------------#
    def solve(self, K):
        """Solves for temperature field using Fourier's heat conduction equation."""
        T_free = jax.scipy.linalg.solve(
            K[self.mesh.bc['free'], :][:, self.mesh.bc['free']],
            self.mesh.bc['heat'][self.mesh.bc['free']],  # Heat source instead of force
            assume_a='pos', check_finite=False
        )
        T = jnp.zeros((self.mesh.ndof))
        T = T.at[self.mesh.bc['free']].set(T_free.reshape(-1))  # Updated
        return T

    # -----------------------#
    def computeThermalObjective(self, K, T):
        """Compute total thermal resistance to minimize heat transfer resistance."""
        thermal_resistance = jnp.sum(K * jnp.square(jnp.gradient(T)))
        return thermal_resistance

    # -----------------------#
    def solveTemperatureField(self):
        """
        Solves the temperature distribution based on the given material distribution.
        Returns: Nodal temperature field.
        """
        def assembleK():
            """Assemble the heat transfer matrix."""
            sK = jnp.zeros((self.mesh.numElems, 4, 4))  # 4 DOFs per element for temperature
            for k in self.k0:
                sK += jnp.einsum('e,jk->ejk', self.k0[k], self.Ktemplates[k])

            K = jnp.zeros((self.mesh.ndof, self.mesh.ndof))
            K = K.at[self.mesh.nodeIdx].add(sK.flatten())
            return K

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
