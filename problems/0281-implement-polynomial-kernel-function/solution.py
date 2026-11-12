import numpy as np

def polynomial_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3, gamma: float = 1.0, coef0: float = 1.0) -> float:
    dot_prod = np.dot(x,y)
    return (gamma*dot_prod + coef0)**degree