def poly_term_derivative(c: float, x: float, n: float) -> float:
    power = n-1
    coeff = c*n
    return coeff*(pow(x, power))