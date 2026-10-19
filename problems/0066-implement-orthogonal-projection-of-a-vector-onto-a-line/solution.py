import numpy as np
def orthogonal_projection(v, L):
	v = np.array(v)
	L = np.array(L)
	scal = np.dot(v, L)/np.dot(L, L)
	return scal * L	
