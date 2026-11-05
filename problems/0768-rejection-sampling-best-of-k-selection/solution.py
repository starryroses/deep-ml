import numpy as np
def rejection_sampling_best_of_k(candidates, scores):
    ans = []
    for i in range(len(candidates)):
        max_score_idx = np.argmax(scores[i])
        ans.append(candidates[i][max_score_idx])
    return ans