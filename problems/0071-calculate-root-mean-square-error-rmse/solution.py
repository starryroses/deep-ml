
import numpy as np

def rmse(y_true, y_pred):
	errors = np.mean((y_true-y_pred)**2)
	rmse_res = np.sqrt(errors)
	return round(rmse_res,3)
