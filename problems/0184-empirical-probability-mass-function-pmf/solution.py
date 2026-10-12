def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    freq = {}
    for num in samples:
        freq[num] = freq.get(num,0) +1

    ans = []
    n = len(samples)
    for x in sorted(freq):
        probab = freq[x]/n
        ans.append((x,probab ))
    return ans