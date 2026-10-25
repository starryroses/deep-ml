import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	X = np.concatenate((np.ones((len(X),1)),X), axis=1)
	theta = np.zeros(X.shape[1])
	losses = []
	for i in range(iterations):
		pred = 1/(1+np.exp(-X@theta))
		loss = -np.sum(y*np.log(pred)+(1-y)*np.log(1-pred))
		losses.append(round(loss,4))
		theta = theta-learning_rate*(X.T@(pred-y))
	return np.round(theta, 4).tolist(), losses