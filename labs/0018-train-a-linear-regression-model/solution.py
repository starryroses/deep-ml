# --- cell 1 ---
import numpy as np

def train(X, y, W, b):
    n = len(X)
    lr = 0.01
    for i in range(100):
        preds = X @ W + b
        errors = preds - y
        grad_W = (2 / n) * (X.T @ errors)
        grad_b = (2 / n) * np.sum(errors)
        W = W - lr * grad_W
        b = b - lr * grad_b
    return W,b