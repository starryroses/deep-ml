import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    standardized_data = []
    normalized_data = []

    for i in range(data.shape[1]):
        col = data[:,i]
        maxi = np.max(col)
        mini = np.min(col)
        std = np.std(col)
        mean = np.mean(col)
        std_col = (col-mean)/std
        norm_col = (col-mini)/(maxi-mini)
        standardized_data.append(std_col)
        normalized_data.append(norm_col)

    standardized_data = np.array(standardized_data).T
    normalized_data = np.array(normalized_data).T
    return standardized_data, normalized_data