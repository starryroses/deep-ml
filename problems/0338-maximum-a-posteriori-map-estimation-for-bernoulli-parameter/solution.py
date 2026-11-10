
import numpy as np

def map_estimate_bernoulli(observations: list, alpha: float, beta: float) -> float:
    
    successes = np.sum(observations)
    failures = len(observations) - successes

    # Posterior parameters
    a_post = alpha + successes
    b_post = beta + failures

    # Handle boundary cases
    if a_post == 1 and b_post == 1:
        return 0.5
    elif a_post == 1:
        return 0.0
    elif b_post == 1:
        return 1.0

    map_estimate = (a_post - 1) / (a_post + b_post - 2)
    return round(float(map_estimate), 4)
