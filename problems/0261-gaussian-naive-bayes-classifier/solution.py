import numpy as np

def gaussian_naive_bayes(X_train, y_train, X_test):
    classes = np.unique(y_train)
    parameters = {}
    total_samples = len(y_train)
    for c in classes:
        X_c = X_train[y_train == c]
        prior = len(X_c) / total_samples
        mean = np.mean(X_c, axis=0)
        variance = np.var(X_c, axis=0)
        parameters[c] = {
            "prior": prior,
            "mean": mean,
            "variance": variance
        }
    predictions = []
    for x in X_test:
        log_probabilities = []
        for c in classes:
            prior = parameters[c]["prior"]
            mean = parameters[c]["mean"]
            variance = parameters[c]["variance"]
            log_prior = np.log(prior)
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * variance) +
                ((x - mean) ** 2) / variance
            )
            log_probability = log_prior + log_likelihood
            log_probabilities.append(log_probability)
        predicted_class = classes[np.argmax(log_probabilities)]
        predictions.append(predicted_class)
    return np.array(predictions)