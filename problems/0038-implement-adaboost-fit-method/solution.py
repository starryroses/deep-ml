
import numpy as np
import math

def adaboost_fit(X, y, n_clf):
    n_samples, n_features = np.shape(X)
    w = np.full(n_samples, 1 / n_samples)
    clfs = []

    for _ in range(n_clf):

        best_error = float('inf')
        best_clf = {}

        # Search all features and thresholds
        for j in range(n_features):
            thresholds = np.unique(X[:, j])

            for t in thresholds:

                # Initial polarity
                p = 1

                # Make predictions
                predictions = np.ones(n_samples)
                predictions[X[:, j] < t] = -1

                # Calculate weighted error
                error = np.sum(w[predictions != y])

                # Flip polarity if error > 0.5
                if error > 0.5:
                    p = -1
                    error = 1 - error

                # Keep best classifier (strictly smaller)
                if error < best_error:
                    best_error = error
                    best_clf = {
						'polarity': p,
						'threshold': t,
						'feature_index': j
					}

        # Calculate classifier weight
        alpha = 0.5 * math.log(
            (1 - best_error) / (best_error + 1e-10)
        )

        best_clf['alpha'] = alpha
        clfs.append(best_clf)

        # Get predictions from best stump
        predictions = np.ones(n_samples)
        predictions[X[:, best_clf['feature_index']] < best_clf['threshold']] = -1
        predictions *= best_clf['polarity']

        # Update sample weights
        w *= np.exp(-alpha * y * predictions)

        # Normalize weights
        w /= np.sum(w)

    return clfs
