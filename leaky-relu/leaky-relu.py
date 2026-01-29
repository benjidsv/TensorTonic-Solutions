import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    x = np.array(x, ndmin=1)
    return np.where(x >= 0, x, x * alpha)