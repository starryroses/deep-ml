def detect_contamination(test_examples, train_corpus, n, threshold):
    ngrams = set()

    # Collect all training n-grams
    for doc in train_corpus:
        words = doc.split()
        for i in range(len(words) - n + 1):
            ngrams.add(tuple(words[i:i+n]))

    contaminated = 0

    # Check each test example
    for example in test_examples:
        words = example.split()
        covered = set()

        for i in range(len(words) - n + 1):
            if tuple(words[i:i+n]) in ngrams:
                covered.update(range(i, i+n))

        coverage = len(covered) / len(words) if words else 0

        if coverage >= threshold:
            contaminated += 1

    if not test_examples:
        return 0.0
    return (contaminated / len(test_examples)) * 100


