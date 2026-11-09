import numpy as np

def gaussian_mle(data: np.ndarray) -> tuple:
    mean = np.mean(data)
    var = np.mean((data-mean)**2)
    return mean,var
    