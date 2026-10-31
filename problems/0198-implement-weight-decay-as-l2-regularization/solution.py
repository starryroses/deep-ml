import numpy as np
def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
	updated_parameters = []
	for param, grad, apply_decay in zip(parameters, gradients, apply_to_all):
		param = np.array(param)
		grad = np.array(grad)
		if apply_decay:
			param = param - lr * grad - lr * weight_decay * param
		else:
			param = param - lr * grad
		updated_parameters.append(param.tolist())
	return updated_parameters
