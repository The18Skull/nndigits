import network
import numpy as np

X = np.array([ [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ] ])
y = np.array([ [ 0 ], [ 1 ], [ 1 ], [ 0 ] ])

nn = network.model()
nn.add_layer(2)
nn.add_layer(1, network.activation.sigmoid)
nn.compile(network.optimization.backprop)
nn.train(X, y, 5)

for a in range(2):
	for b in range(2):
		res = nn.predict(np.array([ a, b ]))
		print("%d XOR %d = %d" % (a, b, res))
