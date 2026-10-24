import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    weights = np.array(weights, dtype=float)

    for i in range(n_iter):
        j = i % len(X)
        error = np.dot(X[j], weights) - y[j]
        gradient = 2 * error * X[j]
        weights -= learning_rate * gradient

    return weights.tolist()

