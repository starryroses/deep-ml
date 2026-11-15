import numpy as np

def precision_recall_curve(y_true: list, y_scores: list) -> tuple:
    thresholds = sorted(set(y_scores), reverse=True)
    precisions = []
    recalls = []
    total_positives = sum(y_true)
    for threshold in thresholds:
        tp = 0
        fp = 0
        for i in range(len(y_true)):
            if y_scores[i] >= threshold:
                if y_true[i] == 1:
                    tp += 1
                else:
                    fp += 1
        predicted_positives = tp + fp
        precision = (
            tp / predicted_positives
            if predicted_positives > 0
            else 1.0
        )
        recall = (
            tp / total_positives
            if total_positives > 0
            else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
    return precisions, recalls, thresholds