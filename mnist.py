from os import path

def mult(arr):
	res = 1
	for x in arr:
		res *= x
	return res

def read(filepath):
	if not path.isfile(filepath):
		return None
	with open(filepath, "rb") as f:
		stream = f.read()
	magic = stream[0:4]
	if magic[0] != 0 or magic[1] != 0:
		return None
	datatype = magic[2]
	dims_len = magic[3]
	dims = [ int.from_bytes(stream[4 * (i + 1):4 * (i + 2)], "big") for i in range(dims_len) ]
	field_size = mult(dims[1:])
	res = [ stream[4 * (dims_len + 1) + i * field_size:4 * (dims_len + 1) + (i + 1) * field_size] for i in range(dims[0]) ]
	return res

def open_dataset(images, labels):
	if not path.isfile(images) or not path.isfile(labels):
		return None
	return read(images), read(labels)
