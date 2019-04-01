import numpy as np
from PIL import Image

#np.random.seed(23) # кочегарим рандом

fast = True # учиться или прочитать из файла
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
x = np.array(img, dtype = np.uint8)

# Инициализация параметров слоев
k = 0
X = np.full(((img.width // width) * (img.height // height), width * height + 1), 255, dtype = np.uint8)
for i in range(0, img.height, height):
	for j in range(0, img.width, width):
		X[k, :-1] = x[i:i + 5, j:j + 3].reshape(height * width)
		k += 1
Y = np.repeat(np.eye(vals, dtype = np.uint8), len(X) // vals, axis = 0)

if fast is True:
	syn0 = np.load("one0.npy")
else:
	# Подготавливаем веса для тренировки
	x = X / 255; y = Y.copy()
	syn0 = 2 * np.random.random((height * width + 1, vals)) - 1

	# Перемешиваем
	r = np.arange(len(x))
	np.random.shuffle(r)
	x = x[r]; y = y[r]

	# Тренировка
	print("[STARTED] Training is in progress...")
	for j in range(epochs):
		for i in range(len(x)):
			a1 = np.dot(x[i], syn0)
			l1 = sigmoid(a1)
			if j % (epochs // 10) == 0 and i == 0:
				err = np.amax(np.abs(y[i] - l1))
				print("[%d/%d] Error: %f" % (j, epochs, err))
			l1_delta = h * (y[i] - l1) * sigmoid(a1, True)
			syn0 += np.dot(x[i].reshape((len(x[i]), 1)), l1_delta.reshape((1, len(l1_delta))))
	np.save("one0", syn0) # Сохраняем результат дял потомков
print("[FINISHED] Average error: %f" % (np.amax(np.abs(Y - sigmoid(np.dot(X / 255, syn0))))))

# Предсказание
def predict(X):
	a1 = np.dot(X, syn0)
	l1 = sigmoid(a1)
	return np.argmax(l1)

# Чтение выборки для проверки с шумом
img = Image.open("trainset0.png").convert("L") # идеальные значения каждого символа
X = np.array(img, dtype = np.uint8).reshape((vals, width * height))
X = np.hstack([X, np.full((vals, 1), 255, dtype = np.uint8)])
Y = np.eye(vals, dtype = np.uint8)

# Проверка с шумом
succ = 0
for i in range(vals): # для всех идеальных изображений
	r = np.argmax(Y[i]) # текущий символ
	print("Number %d:" % r)
	for j in range(1, 5, 1): # размер шума (от 1 до 4)
		res = np.empty(15, dtype = np.uint8)
		for k in range(len(res)): # количество экспериментов для каждого изображения
			img = X[i].copy()
			v = np.arange(width * height)
			np.random.shuffle(v); v = v[:j] # выбираем где шуметь
			img[v] = ~img[v] # реверсим выбранные пиксели
			res[k] = predict(img / 255)
		s = len(np.where(res == r)[0])
		succ += s
		print("\tNoise: %d pixels; Accuracy: %f" % (j, s / len(res)))
print("[SUCCESS] %d of %d (%f)" % (succ, vals * 4 * len(res), succ / (vals * 4 * len(res))))
