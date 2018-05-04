
import tensorflow as tf
import numpy as np
import math
import os, sys
from actioncoder import ActionCoder

class AdverCoder(ActionCoder):

	def discrim(self, carry, fake_input, batch_size, conv_stack=2, disc_spec=[32, 64, 128, 256, 512]):
		for dii, dim in enumerate(disc_spec):
			carry = tf.layers.conv2d(carry, dim, 3, strides=1, padding='same')
			carry = tf.nn.relu(carry)
			carry = tf.layers.conv2d(carry, dim, 3, strides=2, padding='same')
			carry = tf.nn.relu(carry)

		cshape = carry.get_shape()
		carry = tf.reshape(carry, [batch_size, cshape[1] * cshape[2] * cshape[3]])

		carry = tf.nn.relu(tf.layers.dense(carry, 1024))
		carry = tf.nn.relu(tf.layers.dense(carry, 1024))
		carry = tf.nn.relu(tf.layers.dense(carry, 1024))

		# [1, 0] for fake, [0, 1] for real
		isfake = tf.layers.dense(carry, 2)

		fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=fake_input, logits=isfake))

		return isfake, fake_loss

	def __init__(self, images, actions, targets, conv_stack=2, filter_size=3):
		self.images = images
		self.targets = targets
		self.actions = actions

		carry = images
		bsize = tf.shape(images)[0]

		carry, encoder_spec = self.encoder_block(carry, conv_stack=conv_stack)

		carry = self.latent_block(carry, self.actions, bsize)

		carry = self.decoder_block(carry, bsize, encoder_spec, conv_stack=conv_stack)

		self.isfake, self.fake_loss = self.discrim(carry, bsize)

		self.guess = tf.nn.sigmoid(carry)
		print(' [*] layer pattern:', encoder_spec[0])
		print(self.guess.get_shape())

		# bsize = tf.shape(self.images)[:3]
		with tf.variable_scope('Reconst-Loss'):
			loss = tf.nn.l2_loss(self.guess - targets) / tf.cast(bsize, tf.float32)

		self.loss = loss

