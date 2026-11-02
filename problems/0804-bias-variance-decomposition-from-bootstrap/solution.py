import numpy as np

def bias_variance_decomp(predictions, y_true):
    predictions = np.array(predictions)
    y_true = np.array(y_true)
    mean_pred = np.mean(predictions, axis=0)
    bias_squared = np.mean((mean_pred - y_true) ** 2)
    variance = np.mean(np.mean((predictions - mean_pred) ** 2, axis=0))
    mse = np.mean(np.mean((predictions - y_true) ** 2, axis=0))
    return {
        'bias_squared': float(bias_squared),
        'variance': float(variance),
        'mse': float(mse)
    }
