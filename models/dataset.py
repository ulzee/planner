
import numpy as np
from random import shuffle, randint
import h5py
import json

class Dataset:
	def __init__(self):
		self.dbhandle = h5py.File('/media/ul1994/ssd1tb/freeway/frames.h5', 'r')
		self.prepare_data()
		self.tmpl = np.load('chicken.npy')


	def prepare_data(self):
		self.inds = [ii for ii in range(len(self.dbhandle['frames']) - 1) if (ii+1) % 64 != 0 and ii % 64 > 4]
		shuffle(self.inds)

	start_inds = None
	def sample_one(self, bsize, ind=None):
		if self.start_inds is None:
			with open('start_inds.json') as fl:
				self.start_inds = json.load(fl)
			with open('last_inds.json') as fl:
				self.last_inds = json.load(fl)
			# self.last_inds = []
			# self.start_inds = []
			# for ii in range(len(self.dbhandle['frames'])):
			# print()
			# import matplotlib.pyplot as plt
			# for ii in range(len(self.dbhandle['frames'])):
			# # for ii in range(10000):
			# 	sized = resize_data(self.dbhandle['frames'][ii])
			# 	# if is_first(sized, self.tmpl):
			# 	# 	self.start_inds.append(ii)
			# 		# plt.figure()
			# 		# plt.imshow(sized)
			# 		# plt.show()
			# 	if is_last(sized, self.tmpl):
			# 		self.last_inds.append(ii)

			# with open('last_inds.json', 'w') as fl:
			# 	json.dump(self.last_inds, fl)
			# assert False
		if ind is not None:
			images = []
			if ind == -1:
				for ii in range(bsize):
					randind = randint(0, len(self.last_inds) - 1)
					frame = self.dbhandle['frames'][self.last_inds[randind]]
					images.append(decolorize(resize_data(frame)))
			else:
				for ii in range(bsize):
					randind = randint(0, len(self.start_inds) - 1)
					frame = self.dbhandle['frames'][self.start_inds[randind] + ind]
					images.append(decolorize(resize_data(frame)))
			return images
		else:
			raise Exception('???')

	def next_batch(self, batch_size):
		ins = []
		actions = []
		outs = []

		binds = self.inds[:batch_size]
		self.inds = self.inds[batch_size:]

		if len(self.inds) < batch_size:
			self.prepare_data()

		for ii in binds:
			before = (resize_data(self.dbhandle['frames'][ii]))
			after = (resize_data(self.dbhandle['frames'][ii+1]))
			before = decolorize(before)
			after = decolorize(after)
			# diff = after - before
			ins.append(before)
			outs.append(after)
			onehot = [0, 0, 0]
			onehot[self.dbhandle['actions'][ii]] = 1
			actions.append(onehot)

		actions = np.array(actions)
		return ins, actions, outs

def resize_data(img):
	canvas = img[13:13+184, 8:, :]
	typed = canvas.astype(np.float32) / 255.0
	return typed

def not_color(pix, tol=0.01):
	return abs(pix[0] - pix[1]) < tol and abs(pix[1] - pix[2]) < tol and abs(pix[0] - pix[2]) < tol

def is_color(pix, palette, tol=0.01):
	for ppp in palette:
		if abs(pix[0] -  ppp[0] / 255) < tol and abs(pix[1] -  ppp[1] / 255) < tol and abs(pix[2] - ppp[2] / 255) < tol:
			return True
	return False

def pad_edges(img, pad=2):
	img[:pad, :] = 170 / 255
	img[-pad:, :] = 170 / 255
	return img

import time
import numpy.linalg as la
def locate_player(img, tmpl):
	pshape = tmpl.shape

	xpos = 35
	minval = 10000000
	miny = -1
	is_black = tmpl[:, :, 0] == 0
	for yy in range(len(img) - pshape[0]):
		patch = img[yy:yy+pshape[0], xpos:xpos+pshape[1], :].copy()
		patch[is_black] = 0
		diff = la.norm(tmpl - patch)

		if diff < minval:
			minval = diff
			miny = yy
	return miny, xpos

def is_first(img, tmpl, tol=3.0):
	pshape = tmpl.shape

	xpos = 35
	minval = 10000000
	is_black = tmpl[:, :, 0] == 0
	endat = len(img) - pshape[0]
	for yy in range(endat - 3, len(img) - pshape[0]):
		patch = img[yy:yy+pshape[0], xpos:xpos+pshape[1], :].copy()
		patch[is_black] = 0
		diff = la.norm(tmpl - patch)

		if diff < minval:
			minval = diff
	return minval < tol

def is_last(img, tmpl, tol=3.0):
	pshape = tmpl.shape

	xpos = 35
	minval = 10000000
	is_black = tmpl[:, :, 0] == 0
	for yy in range(4, 8):
		patch = img[yy:yy+pshape[0], xpos:xpos+pshape[1], :].copy()
		patch[is_black] = 0
		diff = la.norm(tmpl - patch)

		if diff < minval:
			minval = diff
	return minval < tol

def decolorize(frame, nullcolor=240/255):
	changed = frame.copy()
	is_color = np.logical_and(changed != 142/255, changed != 170/255)[:, :, 0]

	is_red = changed[:, :, 0] == 252/255
	is_green = changed[:, :, 1] == 252/255
	is_y3 = changed[:, :, 2] == 84/255
	is_yellow = np.logical_and(np.logical_and(is_red, is_green), is_y3)

	not_chicken = np.logical_and(np.logical_not(is_yellow), is_color)
	changed[not_chicken] = nullcolor

	changed = pad_edges(changed)
	return changed


if __name__ == '__main__':
	import matplotlib.pyplot as plt

	dset = Dataset()

	ins, _, _ = dset.next_batch(1)
	frame = ins[0]

	# print(np.max(frame))

	changed = frame.copy()
	import time
	t0 = time.time()

	nullcolor = 240 / 255
	is_color = np.logical_and(changed != 142/255, changed != 170/255)[:, :, 0]
	# is_color = (changed != 142/255)[:, :, 0]
	is_red = changed[:, :, 0] == 252/255
	is_green = changed[:, :, 1] == 252/255
	is_y3 = changed[:, :, 2] == 84/255
	is_yellow = np.logical_and(np.logical_and(is_red, is_green), is_y3)
	# print(is_yellow.shape)

	not_chicken = np.logical_and(np.logical_not(is_yellow), is_color)
	changed[not_chicken] = nullcolor
	# color_mask = []
	# for row in changed:
	# 	row_mask = []
	# 	for pix in row:
	# 		if not_color(pix):
	# 			row_mask.append([False]*3)
	# 		elif is_color(pix, [[252, 252, 84]]):
	# 			row_mask.append([False]*3)
	# 		else:
	# 			row_mask.append([True]*3)
	# 	color_mask.append(row_mask)
	# print(time.time() - t0)
	# color_mask = np.array(color_mask, dtype=np.bool)
	# changed[color_mask] = 240 / 255

	changed = pad_edges(changed)
	print(time.time() - t0)
	# player = changed[173:183, 99:107, 0] > 0.7
	# pshape = player.shape

	# tmpl = np.zeros((pshape[0], pshape[1], 3))
	# for yy, row in enumerate(player):
	# 	for xx, pix in enumerate(row):
	# 		if player[yy, xx]:
	# 			tmpl[yy, xx] = [1, 1, 0]

	# plt.figure(figsize=(8, 8))
	# plt.imshow(tmpl)
	# plt.show()
	# np.save('chicken.npy', tmpl)


	# plt.figure(figsize=(16, 8))
	# plt.subplot(1, 3, 1)
	# plt.imshow(frame)
	# # plt.subplot(1, 3, 2)
	# # plt.imshow(color_mask.astype(np.float32))
	# plt.subplot(1, 3, 3)
	# plt.imshow(changed)
	# plt.show()
	# assert False

	tmpl = np.load('chicken.npy')
	t0 = time.time()
	py, px = locate_player(changed, tmpl)
	print(time.time() - t0)
	t0 = time.time()
	first = is_first(changed, tmpl)
	print('first', first, time.time() - t0)
	plt.figure(figsize=(14, 10))
	plt.subplot(1, 3, 1)
	plt.imshow(changed[:, 35:43, :])
	plt.subplot(1, 3, 2)
	plt.imshow(changed[py:py+10, px:px+10, :])
	plt.subplot(1, 3, 3)
	plt.imshow(tmpl)
	plt.show()


