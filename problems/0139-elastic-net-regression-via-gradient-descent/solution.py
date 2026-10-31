import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    n,m = X.shape
    weights = np.zeros(m)
    bias = 0.0

    for i in range(max_iter):
       y_pred = X@weights + bias
       error = y_pred - y
       dw = (X.T@error)/n + alpha1*np.sign(weights) +2*alpha2*weights
       db = np.mean(error)
       weights = weights - learning_rate*dw
       bias = bias - learning_rate*db
       if np.linalg.norm(dw)<tol:
           return weights,bias
    return weights,bias



