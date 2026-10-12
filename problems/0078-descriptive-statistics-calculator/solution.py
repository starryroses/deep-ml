import numpy as np
from collections import Counter

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    mean = np.mean(data)
    median = np.median(data)
    mode = Counter(data).most_common(1)[0][0]
    variance = np.var(data)
    std = np.sqrt(variance)
    q25 = np.percentile(data, 25)
    q50 = np.percentile(data, 50)
    q75 = np.percentile(data, 75)

    return {
        "mean" : mean,
        "median" : median,
        "mode": mode,
        "variance": variance ,
        "standard_deviation": std,
        "25th_percentile": q25 ,
        "50th_percentile": q50,
       "75th_percentile":q75,
       "interquartile_range":q75-q25
    }