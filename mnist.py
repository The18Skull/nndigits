import numpy as np
from os import path

def read(filepath):
	if not path.isfile(filepath):
		return None
	with open(filepath, "rb") as f:
		stream = f.read()
	magic = stream[0:4]
	if magic[0] != 0 or magic[1] != 0:
		return None
	datatype = magic[2]; dims_len = magic[3]
	dims = np.array([ int.from_bytes(stream[4 * (i + 1):4 * (i + 2)], "big") for i in range(dims_len) ])
	field_size = np.prod(dims[1:])
	offset = 4 * (dims_len + 1)
	res = np.array([ np.frombuffer(stream[offset + i * field_size:offset + (i + 1) * field_size], dtype = np.uint8) for i in range(dims[0]) ]) # already flat
	return res

def open_dataset(images, labels):
	if not path.isfile(images) or not path.isfile(labels):
		return None
	return read(images), read(labels)
