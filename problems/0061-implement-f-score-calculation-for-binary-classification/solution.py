import numpy as np

def f_score(y_true, y_pred, beta):
	fp=0
	tp=0
	fn=0
	precision=0
	recall=0
	for i in range(len(y_pred)):
		if y_pred[i]==1 and y_true[i]==1:
			tp+=1
		elif y_pred[i]==0 and y_true[i]==1:
			fn+=1
		elif y_pred[i]==1 and y_true[i]==0:
			fp+=1
	precision=float(tp/(tp+fp)) if tp+fp>0 else 0.0
	recall=float(tp/(tp+fn)) if tp+fn>0 else 0.0
	if precision+recall==0:
		return 0.0
	return float(round((1+beta**2)*(precision*recall/((beta**2*precision)+recall)),3))
