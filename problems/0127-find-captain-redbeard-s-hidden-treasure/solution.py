def find_treasure(start_x: float) -> float:
    """Minimizer of f(x) = x^4 - 3x^3 + 2."""
    x = float(start_x)
    for _ in range(50):
        x_new = x - (4 * x - 9) / 4
        if abs(x_new - x) < 1e-12:
            break
        x = x_new
    return x