import numpy as np
from PIL import Image

np.random.seed(23)

def sigmoid(x, der = False):
	sigm = 1.0 / (1.0 + np.exp(-x))
	if der:
		return sigm * (1.0 - sigm)
	return sigm

#X = np.array([ [ 0, 0, 1 ], [ 0, 1, 1 ], [ 1, 0, 1 ], [ 1, 1, 1 ] ])
#y = np.array([ [ 0, 1, 1, 0 ] ]).T

img = Image.open("trainset.png").convert("L")
x = np.array(img) / 255.0

X = np.empty(0)
for i in range(0, img.height, 5):
	for j in range(0, img.width, 3):
		a = x[i:i + 5, j:j + 3].copy().reshape(15)
		if len(X) == 0:
			X = np.concatenate([ X, a ])
		else:
			X = np.vstack([ X, a ])

y = np.empty(0)
for i in range(10):
	a = np.zeros(10)
	a[i] = 1.0
	for _ in range(10):
		if len(y) == 0:
			y = np.concatenate([ y, a ])
		else:
			y = np.vstack([ y, a ])
#y = y.T

syn0 = 2 * np.random.random((15, 10)) - 1

for j in range(60000):
	a1 = np.dot(X, syn0)
	l1 = sigmoid(a1)
	if j % 10000 == 0:
		print("[%d/%d] Error: %f" % (j, 60000, np.max(y - l1)))
	l1_delta = (y - l1) * sigmoid(a1, True)
	syn0 += X.T.dot(l1_delta)
	#print("\n".join([ str(j), str(syn0).replace("\n", ",") ]))
