import numpy as np
def activation_derivatives(x: float) -> dict[str, float]:
    relu_der = 1 if x>0 else 0
    sigmoid = 1/(1+np.exp(-x))
    sigmoid_der = sigmoid*(1-sigmoid)
    tanh = (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))
    tanh_der = 1-tanh**2
    return {
        'sigmoid':sigmoid_der,
        'tanh':tanh_der,
        'relu':relu_der
    }