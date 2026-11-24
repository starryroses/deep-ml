import math

def tanh(x: float) -> float:
    return (math.e**x - math.e**(-x))/(math.e**x + math.e**(-x))