import math

def normal_pdf(x, mean, std_dev):
	val = (1 / (std_dev * math.sqrt(2 * math.pi))) * math.exp(
        -0.5 * ((x - mean) / std_dev) ** 2
    )
    return round(val, 5)