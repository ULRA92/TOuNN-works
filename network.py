import numpy as np
import jax.numpy as jnp
import jax.nn as nn
from jax import jit, random
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

        # JAX compiled forward function
        self.fwdNN = jit(lambda nnwts, x: self.applyNN(nnwts, x.reshape(-1, x.shape[-1]) if x.ndim < 2 else x))

        # Initialize network weights
        _, self.wts = self.init_fn(rand_key, (-1, nnSettings['inputDim']))

    # -----------------------#
    def makeNetwork(self, nnSettings):
        """
        Builds the neural network structure.

        - Uses Swish activation in hidden layers.
        - Last layer outputs (M microstructure types + 1 conductivity value).
        """
        layers = []
        for _ in range(nnSettings['numLayers'] - 1):
            layers.append(stax.Dense(nnSettings['numNeuronsPerLayer']))
            layers.append(Swish)  # Activation function

        # Last layer: Outputs material type + conductivity
        layers.append(stax.Dense(nnSettings['outputDim']))
        return stax.serial(*layers)

    # -----------------------#
    def forward(self, wts, x):
        """
        Forward pass through the neural network.

        - Predicts `mstrType` (microstructure selection).
        - Predicts `kappa` (thermal conductivity field).
        """
        self.wts = wts
        x = x.reshape(-1, x.shape[-1]) if x.ndim < 2 else x  # Ensure correct shape
        nnOut = self.fwdNN(wts, x)

        # Softmax for microstructure selection
        mstrType = nn.softmax(nnOut[:, :-1])

        # Sigmoid for thermal conductivity (scaled between realistic values)
        k_min, k_max = 0.1, 10.0  # Set physical limits for conductivity
        kappa = k_min + (k_max - k_min) * nn.sigmoid(nnOut[:, -1])

        return mstrType, kappa
