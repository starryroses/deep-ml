
import numpy as np

def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0) -> tuple:

    data = np.array(data, dtype=float)
    labels = np.array(labels, dtype=float)

    n_samples = len(data)

    # Initialize alphas and bias
    alphas = np.zeros(n_samples)
    bias = 0.0

    # Precompute kernel matrix
    if kernel == 'linear':
        K = np.dot(data, data.T)

    elif kernel == 'rbf':
        sq_norms = np.sum(data ** 2, axis=1)
        distances = (
            sq_norms[:, None] +
            sq_norms[None, :] -
            2 * np.dot(data, data.T)
        )
        K = np.exp(-distances / (2 * sigma ** 2))

    else:
        raise ValueError("Kernel must be 'linear' or 'rbf'")

    # Pegasos iterations
    for t in range(1, iterations + 1):

        eta = 1 / (lambda_val * t)

        for i in range(n_samples):

            # Calculate decision value using current alphas
            g_i = np.sum(alphas * labels * K[:, i]) + bias

            if labels[i] * g_i < 1:

                alphas[i] += eta * (1 - lambda_val * alphas[i])

                bias += eta * labels[i]

            else:

                alphas[i] -= eta * lambda_val * alphas[i]

    return alphas.tolist(), float(bias)
