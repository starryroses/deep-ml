import numpy as np

def diag(A: np.ndarray) -> np.ndarray:
    out = []
    for i in range(len(A)):
        out.append(A[i][i])
    return np.asarray(out)
