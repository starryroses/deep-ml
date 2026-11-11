def detect_contamination(training_docs: list, benchmark_examples: list, n: int = 8, threshold: float = 0.8) -> dict:
    train_ngrams = set()
    for doc in training_docs:
        words = doc.lower().split()
        for i in range(len(words) - n + 1):
            gram = tuple(words[i:i+n])
            train_ngrams.add(gram)
    scores = []
    flagged = []
    for example in benchmark_examples:
        words = example.lower().split()
        total = len(words) - n + 1
        if total <= 0:
            score = 0.0
        else:
            count = 0
            for i in range(total):
                gram = tuple(words[i:i+n])
                if gram in train_ngrams:
                    count += 1
            score = count / total
        score = round(score, 4)
        scores.append(score)
        if score >= threshold:
            flagged.append(True)
        else:
            flagged.append(False)
    if len(flagged) == 0:
        rate = 0.0
    else:
        rate = flagged.count(True) / len(flagged)
    rate = round(rate, 4)
    return {
        "scores": scores,
        "flagged": flagged,
        "contamination_rate": rate
    }
