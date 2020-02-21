import tensorflow as tf
import tensorflow.contrib.slim as slim


def CNN(net, output):
	print("slim**************")
	with slim.arg_scope([slim.conv2d, slim.fully_connected],
						activation_fn=tf.nn.relu):
		net = slim.conv2d(net, 300, 3, padding='VALID',
						  weights_initializer=tf.contrib.layers.xavier_initializer())
		net = slim.max_pool2d(net, 2, padding='SAME')
		net = slim.conv2d(net, 200, 3, padding='VALID',
						  weights_initializer=tf.contrib.layers.xavier_initializer())
		net = slim.max_pool2d(net, 2, padding='SAME')
		net = slim.flatten(net)
		net = slim.fully_connected(net, 200)
		net = slim.fully_connected(net, 100)
		logits = slim.fully_connected(net, output, activation_fn=None)
	return logits


def DCCNN(patch, spectrum, output):
	pa = tf.layers.conv2d(
		patch,
		filters=200,
		kernel_size=3,
		strides=1,
		padding="same",
		activation=tf.nn.relu,
		name="convLayer"
	)
	pa = tf.layers.max_pooling2d(pa, 2, strides=2, padding="same")

	pa = tf.layers.conv2d(
		pa,
		filters=300,
		kernel_size=3,
		padding="same",
		activation=tf.nn.relu
	)
	pa = tf.layers.max_pooling2d(pa, 2, strides=2, padding="same")

	pa = tf.layers.flatten(pa)

	sp = tf.layers.conv1d(spectrum, filters=30, kernel_size=5, strides=1, padding="valid", activation=tf.nn.relu)
	sp = tf.layers.max_pooling1d(sp, 3, strides=2, padding="valid")
	sp = tf.layers.conv1d(sp, filters=60, kernel_size=5, strides=1, padding="valid", activation=tf.nn.relu)
	sp = tf.layers.max_pooling1d(sp, 3, strides=2, padding="valid")
	sp=tf.layers.flatten(sp)

	net=tf.concat([pa,sp],1)

	net = tf.layers.dense(net, 200, activation=tf.nn.relu)
	net = tf.layers.dense(net, 100, activation=tf.nn.relu)
	net = tf.layers.dense(net, output, activation=None)
	return net
