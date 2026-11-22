import numpy as np
import math
def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    FUNCS = {
        'square': (lambda u: u * u, lambda u: 2 * u),
        'sin':    (math.sin, math.cos),
        'exp':    (math.exp, math.exp),
        'log':    (math.log, lambda u: 1 / u),
    }
    result = 1.0
    for name in reversed(functions):
        f, df = FUNCS[name]
        result *= df(x)
        x = f(x)
    return result