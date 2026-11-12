import numpy as np

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    # Compute squared Euclidean distances
    sq_dist = (
        np.sum(X1**2, axis=1).reshape(-1, 1)
        + np.sum(X2**2, axis=1)
        - 2 * X1 @ X2.T
    )

    # Avoid small negative values due to floating-point errors
    sq_dist = np.maximum(sq_dist, 0)

    # Compute RBF kernel matrix
    return np.exp(-gamma * sq_dist)