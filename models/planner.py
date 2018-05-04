
import os, sys
import tensorflow as tf
import numpy as np
from configs import Default as configs

class Planner:

	def __init__(self, action_dim=3, stack=20, fixed_dim=2, latent_dim=1024, z_dim=100):
		self.latent_states = tf.placeholder(shape=(None, stack, latent_dim), dtype=tf.float32)
		self.known_states = tf.placeholder(shape=(None, stack, fixed_dim), dtype=tf.float32)
		self.target_state = tf.placeholder(shape=(None, latent_dim), dtype=tf.float32)

		self.z_sample = tf.placeholder(shape=(None, z_dim), dtype=tf.float32)

		self.latent_shaped = tf.reshape(self.latent_states, [-1, stack * latent_dim])
		self.known_shaped = tf.reshape(self.known_states, [-1, stack * fixed_dim])

		distrib_spec = [1024, 1024]
		carry = self.z_sample
		with tf.variable_scope('Distrib_Block'):
			for lsize in distrib_spec:
				carry = tf.nn.relu(tf.layers.dense(carry, lsize))
			z_expanded = carry

		self.flattened = tf.concat([
			z_expanded,
			self.latent_shaped,
			self.known_shaped], axis=1)

		print('Flat-in:', self.flattened.get_shape())

		dense_spec = [2048, 2048, 2048, action_dim * stack]
		carry = self.flattened
		with tf.variable_scope('Dense_Block'):
			for lsize in dense_spec:
				carry = tf.nn.relu(tf.layers.dense(carry, lsize))

		with tf.variable_scope('Action_Output'):
			shaped_actions = tf.reshape(carry, [-1, stack, action_dim])
			self.actions_onehot = tf.nn.softmax(shaped_actions)

		print('Onehot:', self.actions_onehot.get_shape())


	def train(self):
		batch_z = np.random.uniform(
			-1, 1,
			[configs.BATCH_SIZE, self.z_dim]).astype(np.float32)