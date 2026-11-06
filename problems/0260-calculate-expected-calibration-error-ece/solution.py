import numpy as np

def expected_calibration_error(y_true, y_prob, n_bins=10):
    n = len(y_true)
    ece = 0
    for i in range(n_bins):
        low = i / n_bins
        high = (i + 1) / n_bins
        indices = [
            j for j in range(n)
            if (low <= y_prob[j] <= high if i == 0
                else low < y_prob[j] <= high)
        ]
        if indices:
            accuracy = sum(y_true[j] for j in indices) / len(indices)
            confidence = sum(y_prob[j] for j in indices) / len(indices)
            ece += (len(indices) / n) * abs(accuracy - confidence)
    return round(ece, 3)

