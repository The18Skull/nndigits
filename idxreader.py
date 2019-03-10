from os import path

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
	imgs = [ stream[16 + i * (dims[1] * dims[2]):16 + (i + 1) * (dims[1] * dims[2])] for i in range(dims[0]) ]
	#for x in enumerate(imgs[55]):
	#	if x[0] % 28 == 0:
	#		print("")
	#	print("%s" % ("#" if x[1] != 0 else "-"), end = "")
