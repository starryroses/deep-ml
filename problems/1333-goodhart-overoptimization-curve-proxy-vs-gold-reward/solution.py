import math
import numpy as np

def goodhart_curve(proxy, gold, fractions):
    proxy = np.array(proxy)
    gold = np.array(gold)
    fractions = np.array(fractions)
    mean_proxy = []
    mean_gold = []
    sorted_indices = np.argsort(-proxy, kind="stable")
    n = len(proxy)
    for f in fractions:
        k = max(1,int(np.ceil(n*f)))
        selected_idx = sorted_indices[:k]
        mean_proxy.append(np.mean(proxy[selected_idx]))
        mean_gold.append(np.mean(gold[selected_idx]))
    
    peak_idx = np.argmax(mean_gold)
    peak_fraction = fractions[peak_idx]
    smallest_idx = np.argmin(fractions)
    overopt_gap = mean_gold[peak_idx] - mean_gold[smallest_idx]
    return {
        "mean_proxy": np.round(mean_proxy, 4).tolist(),
        "mean_gold": np.round(mean_gold, 4).tolist(),
        "peak_fraction": float(peak_fraction),
        "overopt_gap": round(float(overopt_gap), 4)
    }
