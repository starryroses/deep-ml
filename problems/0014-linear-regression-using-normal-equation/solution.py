import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    A = X.T @ X
    b = X.T @ y
    coefficients = np.linalg.solve(A, b)
    return np.round(coefficients, 4).tolist()