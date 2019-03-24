import mnist
import numpy as np

np.random.seed(23) # кочегарим рандом

fast = True # учиться или прочитать из файла
h = 1.0 # скорость обучения
epochs = 5 # количество эпох
width = 28; height = 28 # размеры картинок
vals = 10 # количество вариантов ответов
l2_count = 32 # количество нейронов скрытого слоя

# Функции активации
def sigmoid(x, der = False):
	sigm = 1.0 / (1.0 + np.exp(-x))
	if der:
		return sigm * (1.0 - sigm)
	return sigm

if fast:
	syn0 = np.load("syn0.npy")
	syn1 = np.load("syn1.npy")
else:
	# Инициализация параметров слоев
	X, z = mnist.open_dataset("train_imgs.idx", "train_labels.idx") # Чтение выборки для тренировки

	X = np.hstack((X, np.full((len(X), 1), 255)))
	X = X / 255

	y = np.zeros((len(X), vals))
	for i in range(len(y)):
		y[i][z[i]] = 1

	syn0 = 2 * np.random.random((height * width + 1, l2_count)) - 1
	syn1 = 2 * np.random.random((l2_count, vals)) - 1

	# Тренировка
	err = 1.0
	print("[STARTED] Training is in progress...")
	for j in range(epochs):
		for i in range(len(X)):
			a1 = np.dot(X[i], syn0); l1 = sigmoid(a1)
			a2 = np.dot(l1, syn1); l2 = sigmoid(a2)
			err = 0.99 * err + 0.01 * (np.amax((y[i] - l2) ** 2))
			#print("[%d/%d] Error: %f (%d)" % (j, epochs, err, i))
			if j % (epochs // 10) == 0 and i == 0:
				print("[%d/%d] Error: %f" % (j, epochs, err))
			l2_delta = (y[i] - l2) * sigmoid(a2, True)
			l1_delta = np.dot(l2_delta, syn1.T) * sigmoid(a1, True)
			syn1 += h * np.dot(l1.reshape((len(l1), 1)), l2_delta.reshape((1, len(l2_delta))))
			syn0 += h * np.dot(X[i].reshape((len(X[i]), 1)), l1_delta.reshape((1, len(l1_delta))))
		h /= 2
	np.save("syn0", syn0); np.save("syn1", syn1) # Сохраняем результат дял потомков
	print("[FINISHED] Average error: %f" % (np.mean((y - sigmoid(np.dot(sigmoid(np.dot(X, syn0)), syn1))) ** 2)))

# Тестирование
# Инициализация параметров слоев
X, z = mnist.open_dataset("test_imgs.idx", "test_labels.idx") # Чтение выборки для тренировки

X = np.hstack((X, np.full((len(X), 1), 255)))
X = X / 255

y = np.zeros((len(X), vals))
for i in range(len(y)):
	y[i][z[i]] = 1

print("[TESTS] Average error: %f" % (np.mean((y - sigmoid(np.dot(sigmoid(np.dot(X, syn0)), syn1))) ** 2)))

print("[!] Ready to predict")

# Предсказание
def predict(X):
	a1 = np.dot(X, syn0); l1 = sigmoid(a1)
	a2 = np.dot(l1, syn1); l2 = sigmoid(a2)
	return np.argmax(l2)
