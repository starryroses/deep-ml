import numpy as np
from itertools import product

def grid_search(X_train: np.ndarray, y_train: np.ndarray, 
                X_val: np.ndarray, y_val: np.ndarray,
                param_grid: dict, model_fn: callable, 
                scoring_fn: callable) -> tuple:
    keys = param_grid.keys()
    values = param_grid.values()

    best_score = float('-inf')
    best_params = {}

    for combination in product(*values):
        params = dict(zip(keys, combination))

        y_pred = model_fn(X_train, y_train, X_val, **params)
        score = scoring_fn(y_val, y_pred)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, round(best_score, 4)