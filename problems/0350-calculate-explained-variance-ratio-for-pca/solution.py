import numpy as np

def explained_variance_ratio(X):
    mean = np.mean(X, axis=0)
    X = X-mean
    cov_mat = np.cov(X, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov_mat)
    eignratio = eigenvalues/np.sum(eigenvalues)
    return sorted(eignratio, reverse=True)
