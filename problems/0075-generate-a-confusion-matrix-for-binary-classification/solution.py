
from collections import Counter

def confusion_matrix(data):
	matrix = [[0 for i in range(2)]for j in range(2)]
	tp=0
	fp=0
	tn=0
	fn=0
	for row in data:
		if row[0]==1 and row[1]==1:
			tp+=1
		elif row[0]==1 and row[1]==0:
			fn+=1
		elif row[0]==0 and row[1]==0:
			tn+=1
		elif row[0]==0 and row[1]==1:
			fp+=1
	matrix[0][0] = tp
	matrix[0][1] = fn
	matrix[1][0] = fp
	matrix[1][1] = tn
	return matrix