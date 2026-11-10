import numpy as np

def pca_reconstruction_error(X: np.ndarray, n_components: int) -> float:
    mean = np.mean(X, axis=0)
    X = X-mean
    cov_mat = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:,idx]
    components = eigenvectors[:,:n_components]
    projected = X @ components
    X_reconstructed = projected @ components.T 
    mse = np.mean((X - X_reconstructed)**2)
    return mse
