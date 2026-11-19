import numpy as np


def digit_network_forward(image, weights, biases):
    a = np.asarray(image).flatten()
    for w, b in zip(weights, biases):
        w = np.asarray(w)
        b = np.asarray(b)
        z = w @ a + b
        a = 1 / (1 + np.exp(-z))
    digit = int(np.argmax(a))
    activations = np.round(a, 4).tolist()
    return activations, digit