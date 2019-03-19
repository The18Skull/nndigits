import numpy as np
from PIL import Image

np.random.seed(23)

def sigmoid(x, der = False):
	sigm = 1.0 / (1.0 + np.exp(-x))
	if der:
		return sigm * (1.0 - sigm)
	return sigm

X = np.array([ [ 1, 0 ], [ 0, 1 ], [ 0, 0 ], [ 1, 1 ] ])
y = np.array([ [ 1, 1, 0, 0 ] ]).T

syn0 = 2 * np.random.random((2, 1)) - 1

for j in range(60000):
	a1 = np.dot(X, syn0)
	l1 = sigmoid(a1)
	if j % 10000 == 0:
		print("[%d/%d] Error: %f" % (j, 60000, np.max(np.abs(y - l1))))
		print("\n".join([ str(j), str(syn0).replace("\n", ",") ]))
	l1_delta = 1.0 * (y - l1) * sigmoid(a1, True)
	syn0 += np.dot(X.T, l1_delta)

for a in range(2):
	for b in range(2):
		a1 = np.dot(np.array([ a, b ]), syn0)
		l1 = sigmoid(a1)
		print("%d XOR %d = %f" % (a, b, l1))
