import numpy as np

def batch_iterator(X, y=None, batch_size=64):
    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = start + batch_size
        if y is not None:
            yield X[start:end], y[start:end]
        else:
            yield X[start:end]