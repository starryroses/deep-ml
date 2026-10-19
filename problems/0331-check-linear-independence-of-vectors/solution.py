import numpy as np

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    matrix = np.array(vectors).T
    rank = np.linalg.matrix_rank(matrix)
    return rank == len(vectors)
        