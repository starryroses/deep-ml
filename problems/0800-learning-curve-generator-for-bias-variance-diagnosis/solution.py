import numpy as np

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    X_val = np.array(X_val, dtype=float)
    y_val = np.array(y_val, dtype=float)

    def make_features(X, d):
        x = X.flatten()
        return np.vstack([x ** k for k in range(d + 1)]).T

    train_errors = []
    val_errors = []
    for n in train_sizes:
        Xn = X_train[:n]
        yn = y_train[:n]
        Phi = make_features(Xn, degree)
        w = np.linalg.pinv(Phi) @ yn
        train_pred = Phi @ w
        train_mse = float(np.mean((train_pred - yn) ** 2))
        Phi_val = make_features(X_val, degree)
        val_pred = Phi_val @ w
        val_mse = float(np.mean((val_pred - y_val) ** 2))
        train_errors.append(train_mse)
        val_errors.append(val_mse)

    final_train = train_errors[-1]
    final_val = val_errors[-1]
    gap = final_val - final_train
    if final_train > bias_threshold:
        diagnosis = 'high_bias'
    elif gap > variance_threshold:
        diagnosis = 'high_variance'
    else:
        diagnosis = 'good_fit'

    return {'train_errors': train_errors, 'val_errors': val_errors, 'diagnosis': diagnosis}