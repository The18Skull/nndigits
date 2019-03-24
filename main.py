import network
import numpy as np
from tkinter import *
from tkinter import messagebox
#from PIL import Image

def center_image(img):
	height, width = img.shape # image sizes

	# cut digit from image
	cols = np.where(np.sum(img, axis = 0) > 0)
	rows = np.where(np.sum(img, axis = 1) > 0)

	x1, x2 = cols[0][0], cols[0][-1] + 1
	y1, y2 = rows[0][0], rows[0][-1] + 1

	digit = img[y1:y2, x1:x2]
	cr_height, cr_width = digit.shape # digit sizes

	# now making result
	res = np.zeros((width, height), dtype = np.uint8)
	rcx, rcy = width // 2, height // 2

	x0 = rcx - cr_width // 2
	y0 = rcy - cr_height // 2

	res[y0:y0 + cr_height, x0:x0 + cr_width] = digit

	return res

class app(Tk):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.title("GUI")
		self.resizable(False, False)
		self.cols = 28; self.rows = 28
		self.width = 500; self.height = 530
		self.geometry("%dx%d" % (self.width, self.height))

		w = self.cols * (self.width // self.cols)
		h = self.rows * ((self.height - 30) // self.rows)
		self.canvas = Canvas(self, width = w + 1, height = h + 1, bg = "white")
		self.canvas.config(highlightthickness = False)
		self.canvas.width = w; self.canvas.height = h
		self.clean()
		self.canvas.bind("<Motion>", self.draw)
		self.canvas.pack(side = TOP)

		self.action = Button(self, text = "Определить", width = 34, command = self.predict)
		self.action.pack(side = LEFT)

		self.clear = Button(self, text = "Очистить", width = 34, command = self.clean)
		self.clear.pack(side = RIGHT)

	def draw(self, ev):
		if ev.x < self.canvas.width and ev.y < self.canvas.height:
			x = ev.x - ev.x % (self.canvas.width // self.cols)
			y = ev.y - ev.y % (self.canvas.height // self.rows)
			j = x // (self.canvas.width // self.cols)
			i = y // (self.canvas.height // self.rows)
			if i < self.rows and j < self.cols:
				if ev.state == 264:
					self.arr[i, j] = 255
					self.canvas.create_rectangle(x, y, x + (self.canvas.width // self.cols), y + (self.canvas.height // self.rows), fill = "blue")
				elif ev.state == 1032:
					self.arr[i, j] = 0
					self.canvas.create_rectangle(x, y, x + (self.canvas.width // self.cols), y + (self.canvas.height // self.rows), fill = "white")

	def predict(self):
		#arr = self.arr.reshape(self.rows * self.cols)
		centered = center_image(self.arr)
		#img = Image.fromarray(centered, "L")
		#img.show()
		arr = np.hstack([ centered.reshape(self.rows * self.cols), np.array([ 255 ]) ])
		res = network.predict(arr / 255)
		messagebox.showinfo(title = "Это похоже на...", message = str(res))
	
	def clean(self):
		self.canvas.delete("all")
		self.arr = np.zeros((self.rows, self.cols), dtype = np.uint8)
		for j in range(self.cols):
			for i in range(self.rows):
				x = j * (self.canvas.width // self.cols)
				y = i * (self.canvas.height // self.rows)
				self.canvas.create_rectangle(x, y, x + (self.canvas.width // self.cols), y + (self.canvas.height // self.rows))

if __name__ == "__main__": app().mainloop()