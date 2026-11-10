import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std==0.0] = 1.0
    data = (data-mean)/std
    cov_mat = np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
    indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:,indices]
    principle_components = eigenvectors[:,:k]
    for i in range(principle_components.shape[1]):
        for j in range(principle_components.shape[0]):
            if abs(principle_components[j,i]) > 1e-10:
                if principle_components[j,i] <0:
                    principle_components[:, i] *= -1
                break
    return principle_components