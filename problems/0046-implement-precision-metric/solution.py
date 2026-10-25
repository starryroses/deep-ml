import numpy as np
def precision(y_true, y_pred):
	tp = 0
	fp = 0
	for i in range(len(y_pred)):
		if y_pred[i] == y_true[i] and y_pred[i]==1:
			tp+=1
		elif y_pred[i]==1 and y_true[i]==0:
			fp+=1
	if fp+tp==0:
		return 0.0
	return float(tp/(tp+fp))
