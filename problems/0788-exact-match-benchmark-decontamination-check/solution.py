def check_contamination(benchmark_prompts, training_data):
    ans = []
    for i in range(len(benchmark_prompts)):
        if benchmark_prompts[i] in training_data:
            ans.append(True)
        else:
            ans.append(False)
    return ans