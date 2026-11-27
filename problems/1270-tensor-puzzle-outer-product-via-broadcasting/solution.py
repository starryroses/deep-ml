import numpy as np

def outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a)[:, None] * np.asarray(b)[None, :]
