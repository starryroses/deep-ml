import numpy as np
from typing import Tuple

def calculate_gini(feature):
    _,counts = np.unique(feature, return_counts=True)
    probab = counts/len(feature)
    gini = 1-np.sum(probab**2)
    return gini

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    n,d = X.shape
    best_feature_index = 0
    best_threshold = 0
    best_gini = float('inf')
    for feature in range(d):
        thresholds = np.unique(X[:,feature])
        for threshold in thresholds:
            left = y[X[:,feature]<=threshold]
            right = y[X[:,feature]>threshold]
            left_gini = calculate_gini(left)
            right_gini = calculate_gini(right)
            weighted_gini = (len(left)*left_gini)/len(y) + (len(right)*right_gini)/len(y)
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature_index = feature
                best_threshold = threshold
    return best_feature_index, best_threshold
