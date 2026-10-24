
import numpy as np
from collections import Counter

def dummy_classifier(y_train, n_test, strategy, constant=None):
    classes = sorted(set(y_train))

    if strategy == "most_frequent":
        counts = Counter(y_train)
        label = min(classes, key=lambda c: (-counts[c], c))
        return [label] * n_test

    elif strategy == "constant":
        return [constant] * n_test

    elif strategy == "uniform":
        return [classes[i % len(classes)] for i in range(n_test)]

    elif strategy == "stratified":
        counts = Counter(y_train)
        exact = {c: n_test * counts[c] / len(y_train) for c in classes}
        allocated = {c: int(exact[c]) for c in classes}
        remaining = n_test - sum(allocated.values())

        order = sorted(classes, key=lambda c: (-(exact[c] - allocated[c]), c))
        for c in order[:remaining]:
            allocated[c] += 1

        return [c for c in classes for _ in range(allocated[c])]

    else:
        raise ValueError("Invalid strategy")
