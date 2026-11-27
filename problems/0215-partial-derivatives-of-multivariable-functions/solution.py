import numpy as np

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
    x = point[0]
    y = point[1]
    z = point[2] if len(point)==3 else -1
    if func_name == 'poly2d':
        return 2*x*y + y**2, x**2 + 2*x*y
    elif func_name == 'exp_sum':
        return np.exp(x+y)*(1+y), np.exp(x+y)*(1+x)
    elif func_name == 'product_sin':
        return np.sin(y), x*np.cos(y)
    elif func_name == 'poly3d':
        return 2*x*y, x**2 + z**2, 2*y*z
    elif func_name == 'squared_error':
        return (float(2 * (x - y)), float(-2 * (x - y)))
    else:
        return -1        