import tensorflow as tf


def CNN(x, output):
	net = tf.layers.conv2d(
		x,
		filters=200,
		kernel_size=3,
		strides=1,
		padding="same",
		activation=tf.nn.relu,
		name="convLayer"
	)
	net = tf.layers.max_pooling2d(net, 2, strides=2, padding="same")

	net = tf.layers.conv2d(
		net,
		filters=300,
		kernel_size=3,
		padding="same",
		activation=tf.nn.relu
	)
	net = tf.layers.max_pooling2d(net, 2, strides=2, padding="same")

	net = tf.layers.flatten(net)

	net = tf.layers.dense(net, 200, activation=tf.nn.relu)
	net = tf.layers.dense(net, 100, activation=tf.nn.relu)
	net = tf.layers.dense(net, output, activation=None)

	return net


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
