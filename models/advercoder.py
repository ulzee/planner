
import tensorflow as tf
import numpy as np
import math
import os, sys
from actioncoder import ActionCoder

class AdverCoder(ActionCoder):

	def discrim(self, carry, discrim_label, batch_size, conv_stack=2, disc_spec=[32, 64, 128, 256, 512], reuse=False):
		with tf.variable_scope('Discriminator') as scope:
			if reuse:
				scope.reuse_variables()

			with tf.variable_scope('Conv_Block'):
				for dii, dim in enumerate(disc_spec):
					carry = tf.layers.conv2d(carry, dim, 3, strides=1, padding='same')
					carry = tf.nn.relu(carry)
					carry = tf.layers.conv2d(carry, dim, 3, strides=2, padding='same')
					carry = tf.nn.relu(carry)

			cshape = carry.get_shape()
			carry = tf.reshape(carry, [batch_size, cshape[1] * cshape[2] * cshape[3]])

			with tf.variable_scope('Dense_Block'):
				carry = tf.nn.relu(tf.layers.dense(carry, 1024))
				carry = tf.nn.relu(tf.layers.dense(carry, 1024))

			# [1, 0] for fake, [0, 1] for real
			isfake = tf.layers.dense(carry, 2)

			discrim_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=discrim_label, logits=isfake))

			return discrim_loss

	def __init__(self, images, actions, targets, fake, conv_stack=2, filter_size=3):
		self.images = images
		self.targets = targets
		self.actions = actions
		self.fake = fake

		carry = images
		bsize = tf.shape(images)[0]

		carry, encoder_spec = self.encoder_block(carry, conv_stack=conv_stack)

		carry = self.latent_block(carry, self.actions, bsize)

		carry = self.decoder_block(carry, bsize, encoder_spec, conv_stack=conv_stack)

		self.real_loss = self.discrim(images, fake, bsize)
		self.fake_loss = self.discrim(carry, fake, bsize, reuse=True)

		self.guess = tf.nn.sigmoid(carry)
		print(' [*] layer pattern:', encoder_spec[0])
		print(self.guess.get_shape())

		with tf.variable_scope('Reconst-Loss'):
			loss = tf.nn.l2_loss(self.guess - targets) / tf.cast(bsize, tf.float32)

		self.loss = loss

if __name__ == '__main__':
	import sys, os
	import h5py
	import matplotlib.pyplot as plt
	from random import shuffle
	import cv2

	from dataset import Dataset

	sess = tf.Session()

	input_frame = tf.placeholder(shape=[None, 184, 152, 3], dtype=tf.float32)
	output_frame = tf.placeholder(shape=[None, 184, 152, 3], dtype=tf.float32)
	action_input = tf.placeholder(shape=[None, 1], dtype=tf.float32)
	fake_input = tf.placeholder(shape=[None, 2], dtype=tf.float32)

	model = AdverCoder(input_frame, action_input, output_frame, fake_input)
	loss = model.loss

	tf.summary.image('input', input_frame)
	tf.summary.image('output', model.guess)
	tf.summary.image('target', output_frame)
	tf.summary.scalar('loss', loss)
	tf.summary.scalar('fake_loss', model.fake_loss)
	tf.summary.scalar('real_loss', model.real_loss)

	train_coder = tf.train.AdamOptimizer(0.0001, epsilon=1e-8) \
		.minimize(loss, global_step=tf.train.get_global_step())
	train_real = tf.train.AdamOptimizer(0.0001, epsilon=1e-8) \
		.minimize(model.real_loss, global_step=tf.train.get_global_step())
	train_fake = tf.train.AdamOptimizer(0.0001, epsilon=1e-8) \
		.minimize(model.fake_loss, global_step=tf.train.get_global_step())
	merged = tf.summary.merge_all()

	train_writer = tf.summary.FileWriter('logs/advercoder/train', graph=sess.graph)

	if len(sys.argv) > 1:
		print('Restoring from:', sys.argv[1])
		tf.train.Saver().restore(sess, sys.argv[1])
	else:
		sess.run(tf.global_variables_initializer())

	BATCH_SIZE = 64

	import sys
	import numpy as np

	def coder_dict(dbatch):
		ins, actions, outs = dbatch

		fake_labels = np.zeros((BATCH_SIZE, 2))
		fake_labels[:, 0] = 1

		return {input_frame:ins, action_input:actions, output_frame:outs, fake_input:fake_labels}

	def discrim_dict(dbatch, is_fake=True):
		fake_labels = np.zeros((BATCH_SIZE, 2))
		fake_labels[:, 0 if is_fake else 1] = 1

		if is_fake:
			ins = dbatch
			return {input_frame:ins, action_input:np.zeros((BATCH_SIZE, 1)), output_frame:ins, fake_input:fake_labels}
		else:
			ins, actions, _ = dbatch
			return {input_frame:ins, action_input:actions, output_frame:ins, fake_input:fake_labels}


	dset = Dataset()

	steps = 100000
	for ii in range(steps):
		dbatch = dset.next_batch(BATCH_SIZE)
		feed_dict = coder_dict(dbatch)

		loss_coder, _, reconst_images, summary = sess.run([loss, train_coder, model.guess, merged], feed_dict=feed_dict)
		train_writer.add_summary(summary, ii)

		feed_dict = discrim_dict(reconst_images, is_fake=True)

		loss_fake, _, summary = sess.run([model.fake_loss, train_fake, merged], feed_dict=feed_dict)

		feed_dict = discrim_dict(dbatch, is_fake=False)

		loss_real, _, summary = sess.run([model.real_loss, train_real, merged], feed_dict=feed_dict)

		sys.stdout.write('[%d/%d] E:%.2f, D[R/F]:[%.2f / %.2f]\r ' % (ii+1, steps, loss_coder, loss_real, loss_fake))
		sys.stdout.flush()

	# 	if ii % 100 == 0 and ii != 0:
	# 		tf.train.Saver().save(sess, 'checkpoints/actioncoder/model-latest')
