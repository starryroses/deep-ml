def min_max(x: list[float]) -> list[float]:
    res = []
    maxi = max(x)
    mini = min(x)
    for xi in x:
        res.append((xi-mini)/(maxi-mini))
    return res