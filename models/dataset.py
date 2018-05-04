
import numpy as np
from random import shuffle
import h5py

class Dataset:
	def __init__(self):
		self.dbhandle = h5py.File('/media/ul1994/ssd1tb/freeway/frames.h5', 'r')
		self.prepare_data()

	def prepare_data(self):
		self.inds = [ii for ii in range(len(self.dbhandle['frames']) - 1) if (ii+1) % 64 != 0 and ii % 64 > 4]
		shuffle(self.inds)

	def next_batch(self, batch_size):
		global inds
		ins = []
		actions = []
		outs = []

		binds = self.inds[:batch_size]
		self.inds = self.inds[batch_size:]

		if len(self.inds) < batch_size:
			self.prepare_data()

		def resize_data(img):
			canvas = img[13:13+184, 8:, :]
			typed = canvas.astype(np.float32) / 255.0
			return typed

		for ii in binds:
			before = resize_data(self.dbhandle['frames'][ii])
			after = resize_data(self.dbhandle['frames'][ii+1])
			# diff = after - before
			ins.append(before)
			outs.append(after)
			actions.append(self.dbhandle['actions'][ii])

		actions = np.array(actions).reshape((batch_size, 1)) / 2.0 # range of actions is 0, 1, 2
		return ins, actions, outs
