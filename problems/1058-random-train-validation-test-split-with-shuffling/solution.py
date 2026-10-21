import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    n = data.shape[0]
    rng = np.random.default_rng(seed)
    rowsidx = rng.permutation(n)
    shuffled_data = data[rowsidx]
    train_end = int(n*train_frac)
    val_end = train_end + int(n*validation_frac)
    train = shuffled_data[:train_end]
    val = shuffled_data[train_end:val_end]
    test = shuffled_data[val_end:]
    return [train, val, test]