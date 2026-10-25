import numpy as np

def recall(y_true, y_pred):
    tp=0
    fn=0
    for i in range(len(y_pred)):
        if y_true[i]==1 and y_pred[i]==1:
            tp+=1
        elif y_pred[i]==0 and y_true[i]==1:
            fn+=1
    if tp+fn==0:
        return 0.0
    return float(tp/(fn+tp))

