def stratified_kfold_indices(y, n_splits):
    classes = {}

    for i, label in enumerate(y):
        classes.setdefault(label, []).append(i)

    folds = []

    for k in range(n_splits):
        test = []

        for indices in classes.values():
            size, rem = divmod(len(indices), n_splits)
            start = k * size + min(k, rem)
            end = start + size + (1 if k < rem else 0)
            test.extend(indices[start:end])

        test.sort()
        train = [i for i in range(len(y)) if i not in test]

        folds.append([train, test])

    return folds