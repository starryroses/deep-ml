import numpy as np

def backprop_chain(a0: float, y: float, w: list, b: list) -> dict:
    L = len(w)

    # Forward pass: store a's and z's
    a = [a0]
    z = []
    for l in range(L):
        z_l = w[l] * a[l] + b[l]
        z.append(z_l)
        a.append(1 / (1 + np.exp(-z_l)))

    # Backward pass
    dW = [0.0] * L
    dB = [0.0] * L

    delta = 2 * (a[L] - y)              # dC/da_L
    for l in reversed(range(L)):
        s = a[l + 1]                    # sigmoid(z_l)
        delta = delta * s * (1 - s)     # dC/dz_l
        dW[l] = delta * a[l]            # dz_l/dw_l = a_{l-1}
        dB[l] = delta                   # dz_l/db_l = 1
        delta = delta * w[l]            # dC/da_{l-1}, passed to previous layer

    return {
        'dW': [round(float(v), 6) for v in dW],
        'dB': [round(float(v), 6) for v in dB],
    }