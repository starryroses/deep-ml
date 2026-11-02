import math

def binomial_probability(n: int, k: int, p: float) -> float:
    probability = (math.comb(n, k) * p**k * (1-p)**(n-k))
    return round(probability, 5)