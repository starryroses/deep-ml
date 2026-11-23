import numpy as np

def dense_backward(a_prev: np.ndarray, W: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict:
    z = W @ a_prev + b
    a = 1/(1+np.exp(-z))
    c = np.sum((a-y)**2)
    dC_da = 2.0 * (a - y)              
    da_dz = a * (1.0 - a)              
    delta = dC_da * da_dz
    dW = np.outer(delta, a_prev)       
    db = delta                         
    da_prev = W.T @ delta              
    return {
        'dW': dW.tolist(),
        'db': db.tolist(),
        'da_prev': da_prev.tolist(),
    }