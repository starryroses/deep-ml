import numpy as np

def mae(y_true, y_pred):
    mae = np.mean(abs(y_true-y_pred))
    return mae