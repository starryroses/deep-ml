import numpy as np

def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
	#trace  
	trace = 0
	rows = len(matrix)
	cols = len(matrix[0])
	for i in range(rows):
		for j in range(cols):
			if i==j:
				trace += matrix[i][j]
	
	det = np.linalg.det(matrix)
	return [det, trace]
