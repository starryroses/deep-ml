# --- cell 1 ---
import numpy as np

def impute(X: np.ndarray) -> np.ndarray:
    for j in range(X.shape[1]):
        meanval = np.nanmean(X[:,j])
        for i in range(X.shape[0]):
            if np.isnan(X[i,j]):
                X[i,j] = meanval                
    return X