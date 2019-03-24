import numpy as np
from PIL import Image

np.random.seed(23) # кочегарим рандом
h = 1.0 # скорость обучения
epochs = 50000 # количество эпох
width = 3; height = 5 # размеры картинок
vals = 10 # количество вариантов ответов

# Функция активации
def sigmoid(x, der = False):
	sigm = 1.0 / (1.0 + np.exp(-x))
	if der:
		return sigm * (1.0 - sigm)
	return sigm

# Чтение выборки для тренировки
img = Image.open("trainset.png").convert("L")
x = np.array(img) / 255.0

# Инициализация параметров слоев
X = np.empty(0)
for i in range(0, img.height, height):
	for j in range(0, img.width, width):
		a = x[i:i + 5, j:j + 3].reshape(height * width)
		a = np.hstack([ a, np.array([ 1.0 ]) ])
		if len(X) == 0:
			X = np.concatenate([ X, a ])
		else:
			X = np.vstack([ X, a ])
y = np.repeat(np.eye(vals), len(X) // vals, axis = 0)
syn0 = 2 * np.random.random((height * width + 1, vals)) - 1

# Перемешиваем
r = np.arange(len(X))
np.random.shuffle(r)
X = X[r]; y = y[r]

# Тренировка
for j in range(epochs):
	for i in range(len(X)):
		a1 = np.dot(X[i], syn0)
		l1 = sigmoid(a1)
		if j % (epochs // 10) == 0 and i == 0:
			print("[%d/%d] Error: %f" % (j, epochs, np.max(np.abs(y[i] - l1))))
			#print(str(syn0).replace("\n", ","))
		l1_delta = h * (y[i] - l1) * sigmoid(a1, True)
		syn0 += np.dot(X[i].reshape((len(X[i]), 1)), l1_delta.reshape((1, len(l1_delta))))
print("[FINISHED] Average error: %f" % (np.max(np.abs(y - sigmoid(np.dot(X, syn0))))))

# Предсказание
def predict(X):
	a1 = np.dot(X, syn0)
	l1 = sigmoid(a1)
	return np.argmax(l1)
