import numpy as np
import jax.numpy as jnp


# -----------------------#
def getKMatrixGridMeshTemplates(elemSize, physics):
    """
    Generate element stiffness/conductivity matrix templates for different physics models.

    Parameters:
        elemSize: (dx, dy) - Element size dimensions.
        physics: 'structural' or 'thermal' - Determines which template is generated.

    Returns:
        Dictionary of element stiffness/conductivity matrices.
    """
    dx, dy = tuple(elemSize)  # Ensure elemSize is unpacked correctly

    K_templates = None  # ✅ Ensure variable is always defined

    if physics in ['structural', 'compliantMechanism']:
        # Structural stiffness matrices
        def Knn_00(dx, dy):
            t2 = 1.0 / dx
            t3 = dy * t2 * (1.0 / 3.0)
            t4 = dy * t2 * (1.0 / 6.0)
            return jnp.reshape(jnp.array([
                t3, 0, -t3, 0, -t4, 0, t4, 0,
                -t3, 0, t3, 0, t4, 0, -t4, 0,
                dy * t2 * (-1.0 / 6.0), 0, t4, 0, t3, 0, -t3, 0,
                t4, 0, -t4, 0, -t3, 0, t3, 0
            ]), (8, 8))

        def Knn_11(dx, dy):
            t2 = 1.0 / dy
            t3 = dx * t2 * (1.0 / 6.0)
            t4 = dx * t2 * (1.0 / 3.0)
            return jnp.reshape(jnp.array([
                0, 0, 0, 0, 0, 0, 0, 0,
                0, t4, 0, t3, 0, -t3, 0, -t4,
                0, t3, 0, t4, 0, -t4, 0, -t3,
                0, -t3, 0, -t4, 0, t4, 0, t3,
                0, -t4, 0, -t3, 0, t3, 0, t4
            ]), (8, 8))

        K_templates = {'00': Knn_00(dx, dy), '11': Knn_11(dx, dy)}

    elif physics == 'thermal':
        # Thermal conductivity matrices
        def Knn_00(dx, dy):
            t2 = 1.0 / dx
            t3 = (dy * t2) / 3.0
            t4 = (dy * t2) / 6.0
            return jnp.reshape(jnp.array([
                t3, -t3, -t4, t4,
                -t3, t3, t4, -t4,
                -t4, t4, t3, -t3,
                t4, -t4, -t3, t3
            ]), (4, 4))

        def Knn_11(dx, dy):
            t2 = 1.0 / dy
            t3 = (dx * t2) / 3.0
            t4 = (dx * t2) / 6.0
            return jnp.reshape(jnp.array([
                t3, t4, -t4, -t3,
                t4, t3, -t3, -t4,
                -t4, -t3, t3, t4,
                -t3, -t4, t4, t3
            ]), (4, 4))

        def Knn_01(dx, dy):
            return jnp.reshape(jnp.array([
                1.0 / 2.0, 0.0, -1.0 / 2.0, 0.0,
                0.0, -1.0 / 2.0, 0.0, 1.0 / 2.0,
                -1.0 / 2.0, 0.0, 1.0 / 2.0, 0.0,
                0.0, 1.0 / 2.0, 0.0, -1.0 / 2.0
            ], dtype=jnp.float32), (4, 4))  # Ensure JAX-compatible dtype

        K_templates = {'00': Knn_00(dx, dy), '11': Knn_11(dx, dy), '01': Knn_01(dx, dy)}

    if K_templates is None:
        raise ValueError(f"ERROR: Unsupported physics type '{physics}'! Choose 'structural' or 'thermal'.")

    return K_templates
