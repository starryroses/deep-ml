import numpy as np
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	n = len(vectors)
	m = len(vectors[0])
	ans = [[0.0 for j in range(n)]for i in range(n)]
	means = []
	for i in range(n):
		means.append(float(np.mean(vectors[i])))
	for i in range(n):
		for j in range(n):
			total = 0
			for k in range(m):
				total += (vectors[i][k] - means[i]) * (vectors[j][k] - means[j])
			ans[i][j] = total/(m-1)
	return ans