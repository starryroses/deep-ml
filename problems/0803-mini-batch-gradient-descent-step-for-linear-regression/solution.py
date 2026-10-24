import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    Xb = X[batch_indices]
    yb = y[batch_indices]
    error = Xb @ weights + bias - yb
    dw = 2 * (Xb.T @ error) / len(batch_indices)
    db = 2 * np.mean(error)
    weights = weights - lr * dw
    bias = bias - lr * db
    return np.append(weights, bias)

