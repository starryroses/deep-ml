import math

def factorial(n):
	fact = 1
	if n==0 or n==1:
		return 1.0
	while n>0:
		fact = fact*n
		n = n-1
	return float(fact)

def poisson_probability(k, lam):
	val = lam**k * 2.718**(-lam) / factorial(k)
	return round(val,5)