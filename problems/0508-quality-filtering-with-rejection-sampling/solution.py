import numpy as np

def quality_filter_rejection_sampling(scores: list, threshold: float, n_select: int = None) -> dict:
    scores = np.array(scores)
    initial_selection = []
    for i,score in enumerate(scores):
        if score >= threshold:
            initial_selection.append(i)
    sorted_indices = np.argsort(-np.array(scores)[initial_selection], kind="stable")
    initial_selection = np.array(initial_selection)[sorted_indices]
    final_selection = initial_selection[:n_select]
    
    acceptance_rate = round(len(initial_selection)/len(scores),4)
    mean_quality = round(float(np.mean(scores[final_selection])), 4) if len(final_selection) > 0 else 0.0
    return {
        'accepted_indices':[int(i) for i in final_selection],
        'acceptance_rate':acceptance_rate,
        'mean_quality':mean_quality
    }
