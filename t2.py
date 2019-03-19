import numpy as np
from PIL import Image

np.random.seed(23)


def sigmoid(x, der=False):
    sigm = 1.0 / (1.0 + np.exp(-x))
    if der:
        return sigm * (1.0 - sigm)
    return sigm


img = Image.open("trainset.png").convert("L")
x = np.array(img) / 255.0

X = np.empty(0)
for i in range(0, img.height, 5):
    for j in range(0, img.width, 3):
        a = x[i:i + 5, j:j + 3].copy().reshape(15)
        if len(X) == 0:
            X = np.concatenate([X, a])
        else:
            X = np.vstack([X, a])

#unitCollumn = np.array([1 for i in range(X.shape[0])]).reshape((X.shape[0], 1))
#X = np.hstack([X, unitCollumn])

y = np.empty(0)
for i in range(10):
    a = np.zeros(10)
    a[i] = 1.0
    for _ in range(2):
        if len(y) == 0:
            y = np.concatenate([y, a])
        else:
            y = np.vstack([y, a])

syn0 = 2 * np.random.random((15, 10)) - 1

for j in range(6000):
	for iterNum in range(20):
		trainIdx = np.random.randint(0, 20)

		xRow = X[trainIdx]
		yCollumn = y[trainIdx] 


		a1 = np.dot(xRow, syn0)
		l1 = sigmoid(a1)

		if j % 1000 == 0 and iterNum == 0:
			print("[%d/%d] Error: %f" % (j, 60000, np.max(np.abs(yCollumn - l1))))
			#print(str(syn0).replace("\n", ","))
		l1_delta = 1.0 * (yCollumn - l1) * sigmoid(a1, True)
		doted = np.dot(xRow.reshape((15,1)), l1_delta.reshape((1,10)))
		# syn0 += np.dot(xRow, l1_delta)
		syn0 += doted


def predict(imageFlatten):
    return sigmoid(np.dot(imageFlatten, syn0))
