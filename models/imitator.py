
import tensorflow as tf
import numpy as np

class Imitator:

	def first_planner(self, action_dim, latent_dim, known_dim, stack):
		istack = 1
		self.latent0 = tf.placeholder(shape=(None, istack, latent_dim), dtype=tf.float32, name='latent_0')
		self.known0 = tf.placeholder(shape=(None, istack, known_dim), dtype=tf.float32, name='known_0')

		latent_shaped = tf.reshape(self.latent0, [-1, istack, latent_dim, 1])
		known_shaped = tf.reshape(self.known0, [-1, istack, known_dim, 1])

		joined_vars = tf.concat([
			latent_shaped,
			known_shaped], axis=2)

		carry = joined_vars
		conv_spec = [32, 64, 128, 256, 512, 512]
		with tf.variable_scope('Conv_Block'):
			for dsize in conv_spec:
				# carry = tf.layers.conv1d(carry, dsize, (3,), strides=2, padding='same')
				carry = tf.layers.conv2d(carry, dsize, (3, 3), strides=2, padding='same')
				carry = tf.nn.relu(carry)

			conv_flat = tf.reshape(carry, [-1, carry.get_shape()[1] * carry.get_shape()[2] * carry.get_shape()[3]])
			# print('Conv-out:', conv_flat.get_shape())


		with tf.variable_scope('Distrib_Block'):
			distrib_spec = [1024, 512, 256]
			carry = self.p_sample
			for lsize in distrib_spec:
				carry = tf.nn.relu(tf.layers.dense(carry, lsize))
			p_expanded = carry
			# print('Samp-out:', p_expanded.get_shape())

		joined_distrib = tf.concat([
			p_expanded,
			conv_flat], axis=1)

		# print('Joined-in:', joined_distrib.get_shape())

		carry = joined_distrib
		with tf.variable_scope('Dense_Block'):
			carry = tf.nn.relu(tf.layers.dense(carry, 2048))
			carry = tf.nn.relu(tf.layers.dense(carry, 1024))
			carry = tf.nn.relu(tf.layers.dense(carry, stack * action_dim))

		with tf.variable_scope('Action_Output'):
			shaped_actions = tf.reshape(carry, [-1, stack, action_dim])
			actions_onehot = tf.nn.softmax(shaped_actions)

			# print('Onehot:', actions_onehot.get_shape())

		return actions_onehot

	def next_planner(self, action_dim, latent_dim, known_dim, stack):
		self.latent_stack = tf.placeholder(shape=(None, stack, latent_dim), dtype=tf.float32, name='latent_stack')
		self.known_stack = tf.placeholder(shape=(None, stack, known_dim), dtype=tf.float32, name='known_stack')

		latent_shaped = tf.reshape(self.latent_stack, [-1, stack, latent_dim, 1])
		known_shaped = tf.reshape(self.known_stack, [-1, stack, known_dim, 1])

		joined_vars = tf.concat([
			latent_shaped,
			known_shaped], axis=2)

		carry = joined_vars
		conv_spec = [32, 64, 128, 256, 512, 512]
		with tf.variable_scope('Conv_Block') as scope:
			scope.reuse_variables()

			for dsize in conv_spec:
				carry = tf.layers.conv2d(carry, dsize, (3, 3), strides=2, padding='same')
				carry = tf.nn.relu(carry)

			conv_flat = tf.reshape(carry, [-1, carry.get_shape()[1] * carry.get_shape()[2] * carry.get_shape()[3]])
			# print('Conv-out:', conv_flat.get_shape())

		with tf.variable_scope('Distrib_Block') as scope:
			scope.reuse_variables()

			distrib_spec = [1024, 512, 256]
			carry = self.p_sample
			for lsize in distrib_spec:
				carry = tf.nn.relu(tf.layers.dense(carry, lsize))
			p_expanded = carry
			# print('Samp-out:', p_expanded.get_shape())

		joined_distrib = tf.concat([
			p_expanded,
			conv_flat], axis=1)

		# print('Joined-in:', joined_distrib.get_shape())

		carry = joined_distrib
		with tf.variable_scope('Dense_Block_STK'):
			carry = tf.nn.relu(tf.layers.dense(carry, 2048))
			carry = tf.nn.relu(tf.layers.dense(carry, 1024))
			carry = tf.nn.relu(tf.layers.dense(carry, stack * action_dim))

		with tf.variable_scope('Action_Output_STK'):
			shaped_actions = tf.reshape(carry, [-1, stack, action_dim])
			actions_onehot = tf.nn.softmax(shaped_actions)

			# print('Onehot:', actions_onehot.get_shape())

		return actions_onehot

	def discrim(self, actions, stack=20, latent_dim=1024, action_dim=3, reuse=False):
		with tf.variable_scope('Discrim_Block') as scope:
			if reuse:
				scope.reuse_variables()

			latents_shaped = tf.reshape(self.discrim_latents, [-1, stack, latent_dim, 1])
			carry = latents_shaped

			with tf.variable_scope('Conv_Latent'):
				disc_spec = [32, 64, 128, 256, 512]
				for dii, dim in enumerate(disc_spec):
					carry = tf.layers.conv2d(carry, dim, 3, strides=1, padding='same')
					carry = tf.nn.relu(carry)
					carry = tf.layers.conv2d(carry, dim, 3, strides=2, padding='same')
					carry = tf.nn.relu(carry)

			cshape = carry.get_shape()
			conv_flat = tf.reshape(carry, [-1, cshape[1] * cshape[2] * cshape[3]])
			ashape = actions.get_shape()
			action_flat = tf.reshape(actions, [-1, ashape[1] * ashape[2]])
			joined = tf.concat([conv_flat, action_flat], axis=1)

			with tf.variable_scope('Dense_Block'):
				carry = joined
				carry = tf.nn.relu(tf.layers.dense(carry, 2048))
				carry = tf.nn.relu(tf.layers.dense(carry, 1024))

			# [1, 0] for fake, [0, 1] for real
			with tf.variable_scope('Discrim_Loss'):
				fake_guess = tf.layers.dense(carry, 2)
				discrim_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.discrim_label, logits=fake_guess))

		return discrim_loss

	def __init__(self, action_dim=3, stack=20, known_dim=1, latent_dim=1024, p_dim=100):
		self.p_sample = tf.placeholder(shape=(None, p_dim), dtype=tf.float32, name='p_sample')

		self.first_actions_onehot = self.first_planner(action_dim, latent_dim, known_dim, stack)
		self.next_actions_onehot = self.next_planner(action_dim, latent_dim, known_dim, stack)

		self.discrim_latents = tf.placeholder(shape=(None, stack, latent_dim), dtype=tf.float32, name='discrim_latents')
		self.discrim_label = tf.placeholder(shape=(None, 2), dtype=tf.float32, name='discrim_label')

		self.discrim_fake = self.discrim(self.next_actions_onehot, stack, latent_dim, action_dim)

		self.real_actions = tf.placeholder(shape=(None, stack, action_dim), dtype=tf.float32, name='real_actions')
		self.discrim_real = self.discrim(self.real_actions, stack, latent_dim, action_dim, reuse=True)

if __name__ == '__main__':
	from dataset import Dataset
	import matplotlib.pyplot as plt
	from actioncoder import unroll_actions, ActionCoder
	import sys

	BATCH_SIZE = 32
	P_DIM = 100
	STACK_SIZE = 20
	LATENT_SIZE = 1024

	dset = Dataset()
	sess = tf.Session()

	coder = ActionCoder(
		tf.placeholder(shape=[None, 184, 152, 3], dtype=tf.float32),
		tf.placeholder(shape=[None, 3], dtype=tf.float32),
		is_eval=True)
	coder_vars = tf.trainable_variables()

	model = Imitator(latent_dim=LATENT_SIZE, p_dim=P_DIM, stack=STACK_SIZE)

	train_real = tf.train.AdamOptimizer(0.001, epsilon=1e-8) \
		.minimize(model.discrim_real, global_step=tf.train.get_global_step())
	train_fake = tf.train.AdamOptimizer(0.001, epsilon=1e-8) \
		.minimize(model.discrim_fake, global_step=tf.train.get_global_step())

	srl = tf.summary.scalar('real_loss', model.discrim_real)
	sfl = tf.summary.scalar('fake_loss', model.discrim_fake)

	# merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter('logs/imitator/train', graph=sess.graph)
	sess.run(tf.global_variables_initializer())
	tf.train.Saver(coder_vars).restore(sess, 'results/actioncoder_small/model-latest')

	# values = sess.run(variables_names)
	# for k, v in zip(variables_names, values):
	# 	print("Variable: ", k)
	# 	print("Shape: ", v.shape)
	# 	# print(v)
	# assert False

	steps = 1000000
	for eii in range(steps):

		# Generate imagined plans
		loss_fake = 0.0
		frame0 = dset.sample_one(BATCH_SIZE, ind=1)

		latent0 = sess.run(coder.inner_z, feed_dict={coder.images:frame0, coder.actions:np.zeros((BATCH_SIZE, 3))})
		latent0 = latent0.reshape((BATCH_SIZE, 1, LATENT_SIZE))
		known0 = np.zeros((BATCH_SIZE, 1, 1))
		batch_p = np.random.uniform(-1, 1, [BATCH_SIZE, P_DIM]).astype(np.float32)

		in0 = {model.latent0:latent0, model.known0:known0, model.p_sample:batch_p}
		guessed_actions = sess.run(model.first_actions_onehot, feed_dict=in0)

		latents, spatials = unroll_actions(sess, coder, frame0, guessed_actions, stack_size=STACK_SIZE)

		latents = np.array([latent0[:, 0, :]] + latents)
		spatials = np.array([frame0] + spatials)

		assert len(spatials) == STACK_SIZE
		assert len(latents) == STACK_SIZE

		latents = np.swapaxes(latents, 0, 1)
		knowns = np.zeros((BATCH_SIZE, STACK_SIZE, 1))
		in_stack = {model.latent_stack:latents, model.known_stack:knowns, model.p_sample:batch_p}
		# in_stack = { **in_stack, **in0 }
		filtered_actions = sess.run(model.next_actions_onehot, feed_dict=in_stack)


		# Discriminiate between real and fake plans
		fake_label = np.zeros((BATCH_SIZE, 2))
		fake_label[:, 0] = 1
		fake_dict = {model.discrim_latents: latents, model.discrim_label:fake_label}
		fake_dict = {**in_stack, **fake_dict}
		loss_fake = sess.run(model.discrim_fake, feed_dict=fake_dict)

		real_frames, real_actions = dset.sample_sequence(BATCH_SIZE, stack=20)

		real_latents = []
		for bb in range(BATCH_SIZE):
			lout = sess.run(
				coder.inner_z,
				feed_dict={coder.images:real_frames[bb], coder.actions:real_actions[bb]})
			real_latents.append(lout)
		real_latents = np.array(real_latents)
		real_label = np.zeros((BATCH_SIZE, 2))
		real_label[:, 1] = 1
		# print(real_actions[0])
		real_dict = {
			model.discrim_latents: real_latents,
			model.real_actions: real_actions,
			model.discrim_label:real_label
		}
		loss_real, summ = sess.run([model.discrim_real, tf.summary.merge([srl])], feed_dict=real_dict)
		train_writer.add_summary(summ, eii)

		sys.stdout.write('[%d/%d] LF:%.4f  LR:%.4f\r ' % (eii+1, steps, loss_fake, loss_real))
		sys.stdout.flush()

		# if ii % 100 == 0 and ii != 0:
		# 	tf.train.Saver().save(sess, 'checkpoints/actioncoder/model-latest')
	print()





