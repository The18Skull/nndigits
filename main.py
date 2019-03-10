import mnist

imgs, ans = mnist.open_dataset("train_imgs.idx", "train_labels.idx")

with open("out.txt", "w") as f:
	for img in enumerate(imgs[:50]):
		for x in enumerate(img[1]):
			if x[0] % 28 == 0:
				f.write("\n\n")
			f.write("%s" % ("#" if x[1] != 0 else "-"))
		print(int.from_bytes(ans[img[0]], "big"), end = ", ")