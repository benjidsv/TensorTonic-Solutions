import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x = np.array(x, ndmin=1)
    return np.maximum(0, x)