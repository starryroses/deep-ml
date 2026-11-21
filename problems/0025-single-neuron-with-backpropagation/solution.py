import numpy as np

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray,
                 initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    weights = np.array(initial_weights, dtype=float)
    bias = float(initial_bias)
    n = len(labels)
    mse_values = []

    for _ in range(epochs):
        # forward pass
        z = features @ weights + bias
        preds = 1 / (1 + np.exp(-z))

        # loss (computed before the update)
        mse_values.append(round(float(np.mean((preds - labels) ** 2)), 4))

        # backward pass
        dz = 2 * (preds - labels) * preds * (1 - preds) / n
        weights = weights - learning_rate * (features.T @ dz)
        bias = bias - learning_rate * np.sum(dz)

    return np.round(weights, 4), round(bias, 4), mse_values