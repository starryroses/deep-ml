import numpy as np

def accuracy_score(y_true, y_pred):
	total = len(y_pred)
	true_count = 0
	for i in range(total):
		if y_pred[i] == y_true[i]:
			true_count +=1
	return float(true_count/total) if total>0 else -1
