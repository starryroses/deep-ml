import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    scores = np.array(scores)
    denomenator = np.sum(np.exp(scores-max(scores)))
    res = np.exp(scores-max(scores))/denomenator
    return res