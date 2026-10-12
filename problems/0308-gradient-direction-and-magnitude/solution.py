import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	#magnitude
	sum_num=0
	for num in gradient:
		sum_num += pow(num,2)
	magnitude = np.sqrt(sum_num)

	#direction
	direction = []
	for num in gradient:
		if num==0.0:
			direction.append(0.0)
			continue
		direction.append(num/magnitude)
	
	#descent
	descent = []
	for num in direction:
		if num==0.0:
			descent.append(0.0)
			continue
		descent.append(-num)

	return {'magnitude':magnitude, 'direction':direction, 'descent_direction':descent}
