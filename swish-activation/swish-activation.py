import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x = np.array(x, ndmin=1)
    return x * 1 / (1 + np.exp(-x))
