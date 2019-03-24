import numpy as np
from tkinter import *
from tkinter import messagebox
import network_one as network

class app(Tk):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.title("GUI")
		self.resizable(False, False)
		self.cols = 3; self.rows = 5
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
					self.arr[i, j] = 1.0
					self.canvas.create_rectangle(x, y, x + (self.canvas.width // self.cols), y + (self.canvas.height // self.rows), fill = "blue")
				elif ev.state == 1032:
					self.arr[i, j] = 0.0
					self.canvas.create_rectangle(x, y, x + (self.canvas.width // self.cols), y + (self.canvas.height // self.rows), fill = "white")

	def predict(self):
		#arr = self.arr.reshape(self.rows * self.cols)
		arr = np.hstack([ self.arr.reshape(self.rows * self.cols), np.array([ 1.0 ]) ])
		res = network.predict(arr)
		messagebox.showinfo(title = "Это похоже на...", message = str(res))
	
	def clean(self):
		self.canvas.delete("all")
		self.arr = np.zeros((self.rows, self.cols))
		for j in range(self.cols):
			for i in range(self.rows):
				x = j * (self.canvas.width // self.cols)
				y = i * (self.canvas.height // self.rows)
				self.canvas.create_rectangle(x, y, x + (self.canvas.width // self.cols), y + (self.canvas.height // self.rows))

if __name__ == "__main__": app().mainloop()
