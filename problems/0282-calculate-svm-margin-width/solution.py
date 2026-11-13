import numpy as np

def svm_margin_width(w: np.ndarray) -> float:
    w2 = np.linalg.norm(w)
    return 2/w2