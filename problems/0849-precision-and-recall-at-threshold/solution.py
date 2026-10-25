import numpy as np

def precision_recall_at_threshold(y_true, y_scores, threshold):
    preds = []
    for score in y_scores:
        val = 1 if score>threshold else 0
        preds.append(val)
    tp=0
    fp=0
    fn=0
    for i in range(len(y_true)):
        if y_true[i]==1 and preds[i]==1:
            tp+=1
        elif y_true[i]==0 and preds[i]==1:
            fp+=1
        elif y_true[i]==1 and preds[i]==0:
            fn+=1
    precision = float(round(tp/(tp+fp), 4)) if tp+fp>0 else 0.0
    recall = float(round(tp/(tp+fn), 4)) if tp+fn>0 else 0.0
    return [precision, recall]
