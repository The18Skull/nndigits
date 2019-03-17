#import mnist
import numpy as np
from PIL import Image

def read(filename):
	img = Image.open(filename).convert("L")
	arr = np.array(img)
	res = list()
	for y in range(0, img.height, 5):
		for x in range(0, img.width, 3):
			i = arr[y:y + 5, x:x + 3].reshape((1, 15)).copy()
			res.append(i)
	return res

#np.random.seed(23)
dataset = read("trainset.png")
answers = np.arange()
print(dataset)
#imgs, ans = mnist.open_dataset("train_imgs.idx", "train_labels.idx")
#imgs = [ [ x / 255 for x in img ] for img in imgs ] # norm

# with open("out.txt", "w") as f:
# 	for img in enumerate(imgs[:50]):
# 		for x in enumerate(img[1]):
# 			if x[0] % 28 == 0:
# 				f.write("\n\n")
# 			f.write("%s" % ("#" if x[1] != 0 else "-"))
# 		print(int.from_bytes(ans[img[0]], "big"), end = ", ")