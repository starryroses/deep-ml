import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing

    def forward(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        self.log_prior = {}
        self.log_prob_1 = {}
        self.log_prob_0 = {}

        for c in self.classes:
            X_c = X[y == c]
            n_c = len(X_c)

            # Class prior probability
            self.log_prior[c] = np.log(n_c / n_samples)

            # Probability of feature being 1 (Laplace smoothing)
            p = (np.sum(X_c, axis=0) + self.smoothing) / (
                n_c + 2 * self.smoothing
            )

            # Store log probabilities
            self.log_prob_1[c] = np.log(p)
            self.log_prob_0[c] = np.log(1 - p)

    def predict(self, X):
        X = np.asarray(X)

        predictions = []

        for x in X:
            scores = []

            for c in self.classes:
                # Bernoulli likelihood
                log_likelihood = np.sum(
                    x * self.log_prob_1[c] +
                    (1 - x) * self.log_prob_0[c]
                )

                # Posterior log probability
                score = self.log_prior[c] + log_likelihood
                scores.append(score)

            predictions.append(self.classes[np.argmax(scores)])

        return np.array(predictions)