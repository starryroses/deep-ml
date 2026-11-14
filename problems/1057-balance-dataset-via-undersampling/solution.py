def balance_undersample(data: list) -> list:
    if len(data) == 0:
        return []
    counts = {}
    for sample, label in data:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    min_count = min(counts.values())
    result = []
    kept = {}
    for sample, label in data:
        if label not in kept:
            kept[label] = 0
        if kept[label] < min_count:
            result.append((sample, label))
            kept[label] += 1
    return result

