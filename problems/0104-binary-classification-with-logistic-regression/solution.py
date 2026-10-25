import numpy as np

def sigmoid(x):
	return 1 if 1/(1+np.exp(-x)) >= 0.5 else 0 

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	res = []
	for row in X:
		pred = sigmoid(np.dot(row, weights) + bias)
		res.append(pred)
	return np.array(res)

