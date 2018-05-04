
import tensorflow as tf
from model import Model

sess = tf.Session()

model = Model()

train_writer = tf.summary.FileWriter('logs/train', graph=sess.graph)

