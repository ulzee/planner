
import tensorflow as tf
import numpy as np
import math
import os, sys
import sys, os
import h5py
import matplotlib.pyplot as plt
from random import shuffle
import cv2

from models.dataset import Dataset
from models.actioncoder import ActionCoder

if __name__ == '__main__':

	sess = tf.Session()

	input_frame = tf.placeholder(shape=[None, 184, 152, 3], dtype=tf.float32)
	output_frame = tf.placeholder(shape=[None, 184, 152, 3], dtype=tf.float32)
	action_input = tf.placeholder(shape=[None, 3], dtype=tf.float32)

	model = ActionCoder(input_frame, action_input, output_frame)
	tf.train.Saver().restore(sess, 'results/actioncoder_small/model-latest')

	dset = Dataset()

	bsize = 4
	starts = dset.sample_one(bsize, ind=30)

	actions = np.zeros((bsize, 3))
	actions[:, 1] = 1
	ups = sess.run(model.guess, feed_dict={input_frame:starts, action_input:actions})

	actions = np.zeros((bsize, 3))
	actions[:, 2] = 1
	downs = sess.run(model.guess, feed_dict={input_frame:starts, action_input:actions})


	joined = np.concatenate([starts, ups, downs], axis=2)

	print(joined.shape)

	plt.figure(figsize=(14, 8))
	for ii in range(bsize):
		plt.subplot(2, 2, ii+1)
		joined[ii][:, 152] = 0
		joined[ii][:, 152*2] = 0
		plt.imshow(joined[ii])
	plt.tight_layout()
	plt.show()