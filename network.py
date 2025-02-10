import numpy as np
import jax.numpy as jnp
import jax.nn as nn
from jax import random, jit
from jax.example_libraries import stax

np.random.seed(0)
rand_key = random.PRNGKey(0)  # Reproducibility

# -----------------------#
def elementwise(fun, **fun_kwargs):
    """Layer that applies a scalar function elementwise on its inputs."""
    init_fun = lambda rng, input_shape: (input_shape, ())
    apply_fun = lambda params, inputs, **kwargs: fun(inputs, **fun_kwargs)
    return init_fun, apply_fun

Swish = elementwise(nn.swish)  # Activation function

# -----------------------#
class TopNet:
    def __init__(self, nnSettings):
        """
        Initializes the neural network for heat transfer optimization.

        - nnSettings: Dictionary containing neural network hyperparameters.
        """
        self.nnSettings = nnSettings
        self.init_fn, self.applyNN = self.makeNetwork(nnSettings)

        # Initialize network weights
        _, self.wts = self.init_fn(rand_key, (-1, nnSettings['inputDim']))

    def forward(self, wts, x):
        """Forward pass through the neural network."""
        self.wts = wts

        # Reshape `x` to ensure it has the correct dimensions
        if x.ndim == 1:
            x = x.reshape(1, -1)  # Convert 1D input to 2D [1, input_dim]
        elif x.ndim == 2:
            pass  # Input is already properly formatted
        else:
            raise ValueError(f"Invalid shape for input `x`: {x.shape}. Expected 1D or 2D array.")

        print(f"DEBUG: x after reshape: {x.shape}")
        print(f"DEBUG: wts type = {type(wts)}")
        print(f"DEBUG: wts lengths = {[len(w) if isinstance(w, (list, tuple)) else 0 for w in wts]}")

        # Validate against the expected input dimension
        expected_input_dim = self.nnSettings['inputDim']
        if x.shape[-1] != expected_input_dim:
            raise ValueError(f"Input dimension mismatch: expected {expected_input_dim} but got {x.shape[-1]}.")

        # Convert `wts` to JAX array format if necessary
        try:
            wts = tuple(jnp.asarray(w) if isinstance(w, (list, np.ndarray)) else w for w in wts)
        except ValueError as e:
            print(f"ERROR: Shape mismatch in wts: {e}")
            raise

        # Propagate data through the network
        nnOut = self.applyNN(wts, x)  # ✅ Ensure correct input format

        # Softmax for microstructure selection
        mstrType = nn.softmax(nnOut[:, :-1])

        # Sigmoid for thermal conductivity (scaled between realistic values)
        k_min, k_max = 0.1, 10.0
        kappa = k_min + (k_max - k_min) * nn.sigmoid(nnOut[:, -1])

        return mstrType, kappa



