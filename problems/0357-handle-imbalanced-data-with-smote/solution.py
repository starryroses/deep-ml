import numpy as np

def smote(X_minority: np.ndarray, n_synthetic: int, k: int = 5) -> np.ndarray:
    n_samples, n_features = X_minority.shape
    k_actual = min(k, n_samples - 1)
    if k_actual == 0 or n_synthetic == 0:
        return np.empty((0, n_features))
    synthetic = []
    for _ in range(n_synthetic):
        i = np.random.randint(0, n_samples)
        xi = X_minority[i]
        distances = np.zeros(n_samples)
        for a in range(n_samples):
            distances[a] = np.linalg.norm(xi - X_minority[a])
        neighbours = np.argsort(distances)[1:k_actual + 1]
        j = np.random.randint(0, k_actual)
        xnn = X_minority[neighbours[j]]
        gap = np.random.random()
        new_sample = xi + gap * (xnn - xi)
        synthetic.append(new_sample)
    return np.array(synthetic)

