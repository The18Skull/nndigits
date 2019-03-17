import numpy as np
np.random.seed(23)

class activation:
	@staticmethod
	def lin(x, der = False):
		if der:
			return 0
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
			return max([ alpha, 1 ])
		return max([ alpha * x, x ])

class optimization:
	@staticmethod
	def backprop(model, y, nu = 0.9):
		pass

class model:
	def __init__(self):
		self.layers = list()
		self.syns = None

	def add_layer(self, neurons, act_func = activation.lin):
		#if len(self.layers) != 0:
		#	self.syns.append(2 * np.random.random((self.layers[-1]["neurons"], neurons)) - 1)
		self.layers.append({ "neurons": neurons, "act": act_func })

	def compile(self, opt_func):
		self.opt_func = opt_func

	def train(self, X, y, epoch = 1):
		if self.syns is None:
			self.syns = [ None ]
			for l in enumerate(self.layers[1:]):
				self.syns.append(2 * np.random.random((self.layers[l[0] - 1]["neurons"], self.layers[l[0]]["neurons"])) - 1)
		sums = list()
		for e in range(epoch):
			sums.clear()
			for i in range(len(self.layers)):
				s = 
				sums.append()
			d = 0
			print("[%d/%d] Error: %f" % (e, epoch, d))

	def predict(self, X):
		for l in 

	def save(self, filename):
		np.save(filename, self.syns)
	
	def load(self, filename):
		self.syns = np.load(filename)
