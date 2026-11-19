import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    features = np.asarray(features)
    labels = np.asarray(labels)
    weights = np.asarray(weights)
    preds = features@weights.T + bias
    probs = 1 / (1 + np.exp(-preds))
    mse = np.mean((probs - labels)**2)
    return probs, mse