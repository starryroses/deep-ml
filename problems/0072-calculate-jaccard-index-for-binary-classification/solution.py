
import numpy as np

def jaccard_index(y_true, y_pred):
	union = 0
	intersection = 0
	for i in range(len(y_pred)):
		if y_pred[i]==1 and y_true[i]==1:
			intersection+=1
		if y_pred[i]==1 or y_true[i]==1:
			union+=1
	if union==0:
		return 0.0
	result = intersection/union
	return round(result, 3)
