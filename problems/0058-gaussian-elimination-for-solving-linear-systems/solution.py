import numpy as np

def gaussian_elimination(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    # Forward elimination with partial pivoting
    for i in range(n):
        pivot_row = i + np.argmax(np.abs(A[i:, i]))

        # Swap rows
        A[[i, pivot_row]] = A[[pivot_row, i]]
        b[[i, pivot_row]] = b[[pivot_row, i]]

        # Check for singular matrix
        if abs(A[i, i]) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular")

        # Eliminate entries below the pivot
        for r in range(i + 1, n):
            factor = A[r, i] / A[i, i]
            A[r, i:] -= factor * A[i, i:]
            b[r] -= factor * b[i]

    # Back substitution
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x