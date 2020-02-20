import tensorflow as tf

def CNN(x,output):
	net = tf.layers.conv2d(
		x,
		filters=100,
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

	net=tf.layers.flatten(net)

	net=tf.layers.dense(net,200,activation=tf.nn.relu)
	net=tf.layers.dense(net,100,activation=tf.nn.relu)
	net=tf.layers.dense(net,output,activation=None)

	return net

