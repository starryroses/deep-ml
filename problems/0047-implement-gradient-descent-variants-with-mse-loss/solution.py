
import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    weights = np.array(weights, dtype=float)

    m = len(y)

    for epoch in range(n_epochs):

        if method == 'batch':
            batches = [(X, y)]

        elif method == 'stochastic':
            batches = [
                (X[i:i+1], y[i:i+1])
                for i in range(m)
            ]

        elif method == 'mini_batch':
            batches = [
                (X[i:i+batch_size], y[i:i+batch_size])
                for i in range(0, m, batch_size)
            ]

        else:
            raise ValueError("Invalid method")

        for X_batch, y_batch in batches:

            # Predictions
            predictions = np.dot(X_batch, weights)

            # Calculate error
            error = predictions - y_batch

            # Calculate MSE gradient
            gradient = (2 / len(y_batch)) * np.dot(X_batch.T, error)

            # Update weights
            weights -= learning_rate * gradient

    return weights
