
import tensorflow as tf
import numpy as np
import math
import os, sys
from encoder import Encoder

class ActionCoder(Encoder):
	def __init__(self, images, actions, targets, conv_stack=2, filter_size=3):
		self.images = images
		self.targets = targets
		self.actions = actions

		carry = images
		bsize = tf.shape(images)[0]

		carry, encoder_spec = self.encoder_block(carry, conv_stack=conv_stack)

		# with tf.variable_scope('Latent_Block'):
		eshape = list(carry.get_shape())
		flat_size = eshape[1] * eshape[2] * eshape[3]
		__shaped_zin = tf.reshape(carry, [bsize, flat_size])
		action_added = tf.concat([__shaped_zin, self.actions], axis=1)
		print(' [*] latent-in:', __shaped_zin.get_shape())


		self.z = tf.nn.relu(tf.layers.dense(inputs=action_added, units=4096))
		print(' [*] latent:', self.z.get_shape())

		zout = tf.nn.relu(tf.layers.dense(inputs=self.z, units=flat_size))
		shaped_zout = tf.reshape(zout, [bsize, eshape[1], eshape[2], eshape[3]])
		carry = shaped_zout
		print(' [*] latent-out:', shaped_zout.get_shape())


		carry = self.decoder_block(carry, bsize, encoder_spec, conv_stack=conv_stack)

		self.guess = tf.nn.sigmoid(carry)
		print(' [*] layer pattern:', encoder_spec[0])
		print(self.guess.get_shape())

		# bsize = tf.shape(self.images)[:3]
		with tf.variable_scope('Reconst-Loss'):
			loss = tf.nn.l2_loss(self.guess - targets) / tf.cast(bsize, tf.float32)

		self.loss = loss

if __name__ == '__main__':
	import sys, os
	import h5py
	import matplotlib.pyplot as plt
	from random import shuffle
	import cv2

	dbhandle = h5py.File('/media/ul1994/ssd1tb/freeway/frames.h5', 'r')

	sess = tf.Session()

	input_frame = tf.placeholder(shape=[None, 184, 152, 3], dtype=tf.float32)
	output_frame = tf.placeholder(shape=[None, 184, 152, 3], dtype=tf.float32)
	action_input = tf.placeholder(shape=[None, 1], dtype=tf.float32)

	model = ActionCoder(input_frame, action_input, output_frame)
	loss = model.loss

	tf.summary.image('input', input_frame)
	tf.summary.image('output', model.guess)
	tf.summary.image('target', output_frame)
	tf.summary.scalar('loss', loss)

	train = tf.train.AdamOptimizer(0.00001, epsilon=1e-8) \
		.minimize(loss, global_step=tf.train.get_global_step(), colocate_gradients_with_ops=True)
	merged = tf.summary.merge_all()

	train_writer = tf.summary.FileWriter('logs/actioncoder/train', graph=sess.graph)

	if len(sys.argv) > 1:
		print('Restoring from:', sys.argv[1])
		tf.train.Saver().restore(sess, sys.argv[1])
	else:
		sess.run(tf.global_variables_initializer())

	BATCH_SIZE = 128

	inds = None
	def prepare_data():
		global inds
		inds = [ii for ii in range(len(dbhandle['frames']) - 1) if (ii+1) % 64 != 0 and ii % 64 > 4]
		shuffle(inds)

	prepare_data()
	print(' [*] Prepared data:', len(inds))

	def next_batch(BATCH_SIZE):
		global inds
		ins = []
		actions = []
		outs = []

		binds = inds[:BATCH_SIZE]
		inds = inds[BATCH_SIZE:]

		if len(inds) < BATCH_SIZE:
			prepare_data()

		def resize_data(img):
			canvas = img[13:13+184, 8:, :]
			typed = canvas.astype(np.float32) / 255.0
			return typed

		for ii in binds:
			ins.append(resize_data(dbhandle['frames'][ii]))
			actions.append(dbhandle['actions'][ii])
			outs.append(resize_data(dbhandle['frames'][ii+1]))

		actions = np.array(actions).reshape((BATCH_SIZE, 1)) / 2.0 # range of actions is 0, 1, 2
		return {input_frame:ins, action_input:actions, output_frame:outs}

	import sys
	import numpy as np

	steps = 100000
	for ii in range(steps):
		feed_dict = next_batch(BATCH_SIZE)

		loss_out, _, summary = sess.run([loss, train, merged], feed_dict=feed_dict)
		train_writer.add_summary(summary, ii)

		sys.stdout.write('[%d/%d] L:%.2f\r ' % (ii+1, steps, loss_out))
		sys.stdout.flush()

		if ii % 100 == 0 and ii != 0:
			tf.train.Saver().save(sess, 'checkpoints/actioncoder/model-latest')


