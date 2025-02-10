import jax.numpy as jnp
import jax
from jax import jit
from functools import partial
import jax.scipy.linalg
from FE_templates import getKMatrixGridMeshTemplates  # Ensure this supports 'thermal' physics


# ✅ Define `jit_objective` as a separate function
@jit
@partial(jit, static_argnums=(0,))  # ✅ Mark `self` as static
def jit_objective(self, k_field):
    """Objective function wrapped in JIT"""
    k_field = jnp.asarray(k_field)
    K = self.assembleK(k_field)
    T = self.solve(K)
    return self.computeThermalObjective(K, T)



# --------------------------- #
class JAXSolver:
    def __init__(self, mesh, material):
        self.mesh = mesh
        self.material = material
        self.Ktemplates = getKMatrixGridMeshTemplates(mesh.elemSize, 'thermal')
        self.k0 = self.material.getThermalConductivityMatrix(self.mesh)

        # ✅ Pass `self` as static in JIT
        self.objectiveHandle = lambda k_field: jit_objective(self, k_field)

    def computeElementMatrix(self, k):
        """Compute the heat transfer element stiffness matrix."""
        element_matrix = jnp.zeros((4, 4))
        for key in self.Ktemplates:
            element_matrix += k * self.Ktemplates[key]
        return element_matrix

    def assembleK(self, k_field):
        """Assemble the global heat transfer matrix K."""
        k_field = jnp.asarray(k_field)
        sK = jnp.zeros((self.mesh.numElems, 4, 4))

        for elem in range(self.mesh.numElems):
            k = k_field[elem]
            sK = sK.at[elem].set(self.computeElementMatrix(k))

        K = jnp.zeros((self.mesh.ndof, self.mesh.ndof))
        iK, jK = map(jnp.array, self.mesh.nodeIdx)

        for elem in range(self.mesh.numElems):
            for i in range(4):
                for j in range(4):
                    K = K.at[iK[elem * 4 + i], jK[elem * 4 + j]].add(sK[elem, i, j])

        return K


    def solve(self, K):
        """Solve for temperature field using Fourier's equation."""
        free_nodes = self.mesh.bc['free']
        heat_source = self.mesh.bc['heat'][free_nodes]

        T_free = jax.scipy.linalg.solve(
            K[free_nodes, :][:, free_nodes],
            heat_source, assume_a='pos', check_finite=False
        )

        T = jnp.zeros((self.mesh.ndof))
        T = T.at[free_nodes].set(T_free.reshape(-1))
        return T

    def computeThermalObjective(self, K, T):
        """Compute total thermal resistance."""
        return jnp.sum(K * jnp.square(jnp.gradient(T)))
