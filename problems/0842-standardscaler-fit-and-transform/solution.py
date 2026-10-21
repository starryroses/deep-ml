import numpy as np

def standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    std_vals = np.std(X_train, axis=0)
    mean_vals = np.mean(X_train, axis=0)
    std_vals[std_vals==0] =1
    X_test = (X_test-mean_vals)/std_vals

    return np.array(X_test)
        
