import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    x = np.asarray(x, dtype=float)
    numerical_grad = np.zeros_like(x)

    for i in range(x.size):
        x_plus = x.copy().ravel()
        x_minus = x.copy().ravel()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon

        f_plus = f(x_plus.reshape(x.shape))
        f_minus = f(x_minus.reshape(x.shape))
        numerical_grad.flat[i] = (f_plus - f_minus) / (2 * epsilon)

    diff = np.linalg.norm(numerical_grad - analytical_grad)
    total = np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad)

    relative_error = 0.0 if total == 0 else diff / total
    return numerical_grad, relative_error