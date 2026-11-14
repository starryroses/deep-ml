import numpy as np

def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    N, M = X.shape
    C = max(y) + 1
    # Add bias column of ones
    bias = np.ones((N, 1))
    X = np.concatenate((bias, X), axis=1)
    # Initialize weights with zeros
    W = np.zeros((C, M + 1))
    losses = []
    for i in range(iterations):
        # Calculate scores
        scores = X @ W.T
        # Calculate softmax probabilities
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Calculate loss
        loss = 0
        for j in range(N):
            loss += -np.log(probs[j, y[j]])
        losses.append(round(loss, 4))
        # Calculate gradient
        gradient = np.zeros((C, M + 1))
        for j in range(N):
            for k in range(C):
                error = probs[j, k]
                if y[j] == k:
                    error -= 1
                gradient[k] += error * X[j]
        # Update weights
        W = W - learning_rate * gradient
    return np.round(W, 4).tolist(), losses
