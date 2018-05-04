import tensorflow as tf
import numpy as np
import math
import os, sys

class Encoder:

	def save(self, sess, withname):
		tf.train.Saver().save(sess, 'checkpoints/model-%s' % withname)

	def encoder_block(self, images, conv_stack=2, n_filters=[1, 32, 64, 128, 256, 512], filter_size=3):
		assert conv_stack >= 1

		encoder = []
		shapes = []

		carry = images
		for layer_i, n_output in enumerate(n_filters[1:]):
			n_input = carry.get_shape().as_list()[3]
			shapes.append(carry.get_shape().as_list())
			with tf.variable_scope('Encoder_%d' % layer_i):
				for _ in range(conv_stack):
					carry = tf.layers.conv2d(
						inputs=carry,
						filters=n_output,
						padding='same',
						strides=[1, 1],
						kernel_size=[filter_size]*2)

				# encoder.append(carry.get_shape().as_list())
				print(' [*] encoder:', carry.get_shape())
				# if layer_i < len(n_filters[1:]) - 1:
					# maxpool all except last
				encoder.append(carry.get_shape().as_list())
				carry = tf.layers.max_pooling2d(
					inputs=carry,
					pool_size=[2, 2],
					strides=2,
					padding='same')
				carry = tf.nn.relu(carry)
		return carry, (encoder, shapes)

	def decoder_block(self, carry, batch_size, encoder_spec, conv_stack=2, filter_size=3):
		encoder, shapes = encoder_spec
		encoder.reverse()
		shapes.reverse()

		for layer_i, shape in enumerate(shapes):
			with tf.variable_scope('Decoder_%d' % layer_i):
				_, _, _, n_input = shapes[layer_i]
				_, _, _, n_output = encoder[layer_i]
				n_output = carry.get_shape().as_list()[-1]
				W = tf.Variable(
					tf.random_uniform([
						filter_size,
						filter_size,
						n_input, n_output],
						-1.0 / math.sqrt(n_input),
						1.0 / math.sqrt(n_input)))
				b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
				carry = tf.nn.conv2d_transpose(
					carry,
					W,
					tf.stack([batch_size, shape[1], shape[2], shape[3]]),
					strides=[1, 2, 2, 1], padding='SAME')
				carry = tf.add(carry, b)

				if layer_i == len(shapes) - 1 and conv_stack == 1:
					print(' [*] norelu')
					pass
				else:
					carry = tf.nn.relu(carry)

				for ss in range(1, conv_stack):
					carry = tf.layers.conv2d(
						inputs=carry,
						filters=n_input,
						padding='same',
						strides=[1, 1],
						kernel_size=[filter_size]*2)
					if layer_i == len(shapes) - 1 and ss == conv_stack - 1:
						print(' [*] norelu')
						# activation = None
					else:
						carry = tf.nn.relu(carry)

				print(' [*] decoder-%d:' % layer_i, carry.get_shape())
		return carry


	def __init__(self, images, targets, conv_stack=2, filter_size=3):
		self.images = images
		self.targets = targets

		carry = images
		bsize = tf.shape(images)[0]

		carry, encoder_spec = self.encoder_block(carry, conv_stack=conv_stack)

		# with tf.variable_scope('Latent_Block'):
		eshape = list(carry.get_shape())
		flat_size = eshape[1] * eshape[2] * eshape[3]
		shaped_zin = tf.reshape(carry, [bsize, flat_size])
		print(' [*] latent-in:', shaped_zin.get_shape())

		self.z = tf.nn.relu(tf.layers.dense(inputs=shaped_zin, units=4096))
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
