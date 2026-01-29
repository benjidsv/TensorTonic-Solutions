import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.array(x, ndmin=1)
    x = x - np.max(x, keepdims=True)
    x = np.exp(x)
    return x / x.sum(axis=-1, keepdims=True)