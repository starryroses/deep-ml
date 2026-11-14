# --- cell 1 ---
import numpy as np
from sklearn.linear_model import RidgeCV

def train(X_train, y_train, X_val, y_val):
    # Fit on train + val: more data helps when n_samples < n_features
    X = np.vstack([X_train, X_val])
    y = np.concatenate([y_train, y_val])

    # 1) Clip every feature to the range seen in training (with a small margin)
    lo, hi = np.percentile(X, 1, axis=0), np.percentile(X, 99, axis=0)
    X_c = np.clip(X, lo, hi)

    # 2) Strongly regularized ridge, alpha picked by built-in CV
    alphas = np.logspace(-1, 4, 30)
    model = RidgeCV(alphas=alphas)
    model.fit(X_c, y)

    # 3) Clip predictions to a plausible target range
    y_lo, y_hi = y.min(), y.max()
    margin = 0.1 * (y_hi - y_lo)

    def predict(X_new):
        X_new = np.clip(np.asarray(X_new), lo, hi)
        return np.clip(model.predict(X_new), y_lo - margin, y_hi + margin)

    return predict