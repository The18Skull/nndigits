import numpy as np
np.random.seed(23)

class activation:
	@staticmethod
	def lin(x, der = False):
		if der:
			return 1.0
		return x

	@staticmethod
	def sigmoid(x, der = False):
		sigm = 1.0 / (1.0 + np.exp(-x))
		if der:
			return sigm * (1.0 - sigm)
		return sigm

	@staticmethod
	def relu(x, der = False):
		return max([ 0, x ])

	@staticmethod
	def leaky_relu(x, der = False, alpha = 0.01):
		if der:
			return max([ alpha, 1.0 ])
		return max([ alpha * x, x ])

class optimization:
	@staticmethod
	def backprop(model, y, nu = 1.0):
		pass

class neuron:
	def __init__(self, act_func):
		self.f = act_func
		self.a = None
		self.e = None

	def act(self):
		self.a = self.f(self.e)

	def sum(self, l0, syn):
		self.e = np.dot(l0, syn)

class layer:
	def __init__(self, neurons, act_func = activation.lin):
		self.neurons = [ neuron(act_func) for _ in range(neurons) ]
		self.syns = None
	
	def sum(self, l0):
		for i in range(len(self.neurons)):
			self.neurons[i].sum(l0, self.syns[i])

class model:
	def __init__(self):
		self.layers = [ None ]

	def add_layer(self, neurons, act_func = activation.lin):
		self.layers.append(layer(neurons, act_func))

	def compile(self, opt_func):
		self.opt_func = opt_func

	def train(self, X, y, epoch = 1):
		if len(self.syns) == 0:
			for i in range(1, len(self.layers)):
				self.syns.append(2 * np.random.random((self.layers[i]["neurons"], self.layers[i - 1]["neurons"])) - 1)
		for e in range(epoch):
			for i in range(len(X)):
				#self.syns[0] = np.repeat(X[i].reshape((1, len(X[i]))), len(self.syns[1]), axis = 0)
				for j in range(len(self.layers) - 1):
					if j == 0: continue
					self.layers[j]["act"](self.layers[j]["res"])
			# sums = list()
			# for i in range(len(self.layers)):
			# 	s = 
			# 	sums.append(s)
			# d = 0
			print("[%d/%d] Error: %f" % (e, epoch, d))

	def predict(self, X):
		pass

	def save(self, filename):
		syns = np.array
		for i in range(len(self.layers)):
			self.layers[i].syns = syns[i]
		np.save(filename, self.syns)
	
	def load(self, filename):
		syns = np.load(filename)
		for i in range(len(self.layers)):
			self.layers[i].syns = syns[i]
