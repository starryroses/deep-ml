
import math
import numpy as np

def cholesky_decomposition(A):
    try:
        matrix = np.asarray(A, dtype=float)

        # Check for empty or invalid input
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return -1

        n, m = matrix.shape

        # Check if square
        if n != m:
            return -1

        # Check for finite values
        if not np.all(np.isfinite(matrix)):
            return -1

        # Check if symmetric
        if not np.allclose(matrix, matrix.T):
            return -1

        lower = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1):
                total = sum(
                    lower[i][k] * lower[j][k]
                    for k in range(j)
                )

                if i == j:
                    value = matrix[i][i] - total

                    # Must be positive for Cholesky decomposition
                    if value <= 0:
                        return -1

                    lower[i][j] = math.sqrt(value)
                else:
                    if lower[j][j] == 0:
                        return -1

                    lower[i][j] = (
                        matrix[i][j] - total
                    ) / lower[j][j]

        return lower

    except (TypeError, ValueError, OverflowError):
        return -1
