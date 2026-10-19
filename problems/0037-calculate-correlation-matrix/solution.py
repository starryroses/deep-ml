import numpy as np

def calculate_correlation_matrix(X, Y=None):
    if Y is None:
        ans = np.corrcoef(X, rowvar=False)
    else:
        ans = np.corrcoef(X.T, Y.T)[:X.shape[1], X.shape[1]:]
    return np.round(ans, 10).tolist()