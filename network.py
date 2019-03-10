import numpy as np

def sigmoid(x, der = False):
	sigm = 1.0 / (1.0 + np.exp(-x))
	if der:
		return sigm * (1.0 - sigm)
	return sigm

def relu(x, der = False):
	return max([ 0, x ])

def leaky_relu(x, alpha, der = False):
	if der:
		return max([ alpha, 1 ])
	return max([ alpha * x, x ])