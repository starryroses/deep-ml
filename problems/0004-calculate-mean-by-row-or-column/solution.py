import numpy as np
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	rows = len(matrix)
	cols = len(matrix[0])
	matrix = np.array(matrix )
	ans = []
	if mode == "column":
		matrix = matrix.T
		for x in matrix :
			ans.append(np.mean(x))
	else:
		for x in matrix :
			ans.append(np.mean( x))
	return ans