
import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
    x = np.zeros_like(b, dtype=float) if x0 is None else np.array(x0, dtype=float)
    r = b - A @ x
    p = r.copy()
    rs = np.dot(r, r)

    for _ in range(n):
        Ap = A @ p
        alpha = rs / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)

        if np.sqrt(rs_new) < tol:
            break

        p = r + (rs_new / rs) * p
        rs = rs_new

    return x
