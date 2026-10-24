
import numpy as np

def r_squared(y_true, y_pred):
    ssr = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    if sst == 0:
        return 1.0 if ssr == 0 else 0.0
    return round(1 - ssr / sst, 3)

