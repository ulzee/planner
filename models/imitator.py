
import tensorflow as tf
import numpy as np

class Imitator:
	def __init__(self, action_dim=3, stack=20, known_dim=1, latent_dim=1024, p_dim=100):
		self.p_sample = tf.placeholder(shape=(None, p_dim), dtype=tf.float32)

		istack = 1
		self.latent0 = tf.placeholder(shape=(None, istack, latent_dim), dtype=tf.float32)
		self.known0 = tf.placeholder(shape=(None, istack, known_dim), dtype=tf.float32)
		# self.target_state = tf.placeholder(shape=(None, latent_dim), dtype=tf.float32)

		latent_shaped = tf.reshape(self.latent0, [-1, istack * latent_dim])
		known_shaped = tf.reshape(self.known0, [-1, istack * known_dim])

		distrib_spec = [1024, 1024]
		carry = self.p_sample
		with tf.variable_scope('Distrib_Block'):
			for lsize in distrib_spec:
				carry = tf.nn.relu(tf.layers.dense(carry, lsize))
			p_expanded = carry

		flattened = tf.concat([
			p_expanded,
			latent_shaped,
			known_shaped], axis=1)

		flattened_shaped = tf.reshape(flattened, [-1, flattened.get_shape()[1], 1])

		print('Flat-in:', flattened_shaped.get_shape())


		carry = flattened_shaped
		conv_spec = [64, 128, 256, 512, 512]
		with tf.variable_scope('Conv_Block_0'):
			for dsize in conv_spec:
				carry = tf.layers.conv1d(carry, dsize, (3,), strides=2, padding='same')
				carry = tf.nn.relu(carry)

			print('Conv-out:', carry.get_shape())

			carry = tf.reshape(carry, [-1, carry.get_shape()[1] * carry.get_shape()[2]])

			carry = tf.nn.relu(tf.layers.dense(carry, 1024))
			carry = tf.nn.relu(tf.layers.dense(carry, 1024))
			carry = tf.nn.relu(tf.layers.dense(carry, stack * action_dim))

		with tf.variable_scope('Action_Output_0'):
			shaped_actions = tf.reshape(carry, [-1, stack, action_dim])
			self.actions_onehot = tf.nn.softmax(shaped_actions)

			print('Onehot:', self.actions_onehot.get_shape())

		# stack_rem = stack - 1
		# self.latent_stack = tf.placeholder(shape=(None, stack_rem, latent_dim), dtype=tf.float32)
		# self.known_stack = tf.placeholder(shape=(None, stack_rem, known_dim), dtype=tf.float32)

		# latent_shaped = tf.reshape(self.latent_stack, [-1, stack_rem * latent_dim])
		# known_shaped = tf.reshape(self.known_stack, [-1, stack_rem * known_dim])

		# flattened = tf.concat([
		# 	p_expanded,
		# 	latent_shaped,
		# 	known_shaped], axis=1)

		# flattened_shaped = tf.reshape(flattened, [-1, flattened.get_shape()[1], 1])

		# print('Flat-in:', flattened_shaped.get_shape())

		# carry = flattened_shaped
		# conv_spec = [64, 128, 256, 512, 512]
		# with tf.variable_scope('Conv_Block_STK'):
		# 	for dsize in conv_spec:
		# 		carry = tf.layers.conv1d(carry, dsize, (3,), strides=2, padding='same')
		# 		carry = tf.nn.relu(carry)

		# 	print('Conv-out:', carry.get_shape())

		# 	carry = tf.reshape(carry, [-1, carry.get_shape()[1] * carry.get_shape()[2]])

		# 	carry = tf.nn.relu(tf.layers.dense(carry, 1024))
		# 	carry = tf.nn.relu(tf.layers.dense(carry, 1024))
		# 	carry = tf.nn.relu(tf.layers.dense(carry, stack * action_dim))

		# with tf.variable_scope('Action_Output_STK'):
		# 	shaped_actions = tf.reshape(carry, [-1, stack, action_dim])
		# 	self.actions_onehot = tf.nn.softmax(shaped_actions)

		# 	print('Onehot:', self.actions_onehot.get_shape())

if __name__ == '__main__':
	from dataset import Dataset
	import matplotlib.pyplot as plt

	dset = Dataset()

	sess = tf.Session()

	BATCH_SIZE = 64
	P_DIM = 100
	model = Imitator(p_dim=P_DIM)

	merged = tf.summary.merge_all()

	train_writer = tf.summary.FileWriter('logs/imitator/train', graph=sess.graph)

	sess.run(tf.global_variables_initializer())


	for eii in range(100000):
		frame0 = dset.sample_one(ind=1)
		# frameF = dset.sample_one(ind=-1)

		# plt.figure()
		# plt.imshow(frameF)
		# plt.show()
		# assert False
		# dbatch = dset.next_batch()

		latent0 = np.zeros((BATCH_SIZE, 1, 1024))
		known0 = np.zeros((BATCH_SIZE, 1, 1))
		batch_p = np.random.uniform(-1, 1, [BATCH_SIZE, P_DIM]).astype(np.float32)

		in0 = {model.latent0:latent0, model.known0:known0, model.p_sample:batch_p}
		guessed_actions = sess.run(model.actions_onehot, feed_dict=in0)

		def readable_actions(traj):
			readable = []
			for step in traj:
				readable.append(int(np.argmax(step)))
			return readable

		print(guessed_actions.shape)
		print(readable_actions(guessed_actions[0]))
		print(readable_actions(guessed_actions[1]))
		print(readable_actions(guessed_actions[2]))

		assert False



