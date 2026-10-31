import numpy as np

def soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)

def l1_regularization_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
    
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0
    
    for i in range(max_iter):
        y_pred = X@weights + bias
        w_temp = weights - learning_rate*(1/n_samples)*X.T@(y_pred-y)
        w_new = np.sign(w_temp) * np.maximum(np.abs(w_temp)-learning_rate*alpha,0)
        bias = bias-learning_rate*(1/n_samples)*np.sum(y_pred-y)
        check = np.linalg.norm(w_new-weights)
        weights=w_new
        if(check<tol):
            return weights,bias
    return weights,bias
    
