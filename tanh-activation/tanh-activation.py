import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.array(x, ndmin=1)
    e_x = np.exp(x)
    e_mx = np.exp(-x)
    return (e_x - e_mx) / (e_x + e_mx)