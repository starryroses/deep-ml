import numpy as np

def fit_polynomial(x, y, degree):
    X = np.vander(x,degree+1,increasing=True)
    coeffs = np.linalg.solve(X.T@X, X.T@y)
    return coeffs.tolist()
