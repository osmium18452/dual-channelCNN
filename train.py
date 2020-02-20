import tensorflow as tf
import numpy as np
from tqdm import tqdm
import time
import os
import argparse
from dataloader import DataLoader
from model import CNN

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", default=50, type=int)
parser.add_argument("-b", "--batch_size", default=100, type=int)
parser.add_argument("-l", "--lr", default=0.001, type=float)
parser.add_argument("-g", "--gpu", default="0")
parser.add_argument("-r", "--ratio", default=0.1, type=float)
parser.add_argument("-a", "--aug",default=0.5,type=float)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
EPOCHS = args.epochs
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size
RATIO = args.ratio
AUGMENT_RATIO=args.aug

pathName = []
pathName.append("./data/Indian_pines.mat")
pathName.append("./data/Indian_pines_gt.mat")
matName = []
matName.append("indian_pines")
matName.append("indian_pines_gt")

dataloader = DataLoader(pathName, matName, 7, RATIO, AUGMENT_RATIO)

trainPatch, trainLabel = dataloader.loadTrainPatchOnly()
testPatch, testLabel = dataloader.loadTestPatchOnly()

# print(np.shape(trainPatch),np.shape(trainLabel))

x = tf.placeholder(shape=[None, dataloader.patchSize, dataloader.patchSize, dataloader.bands], dtype=tf.float32)
y = tf.placeholder(shape=[None, dataloader.numClasses], dtype=tf.float32)

pred = CNN(x, dataloader.numClasses)
softmaxOutput = tf.nn.softmax(pred)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
correctPredictions = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPredictions, "float"))
predictTestLabels = tf.argmax(pred, 1)
init = tf.global_variables_initializer()
print(np.shape(trainPatch), np.shape(trainLabel))
exit(0)

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(EPOCHS):
		if epoch % 5 == 0:
			permutation = np.random.permutation(trainPatch.shape[0])
			trainPatch = trainPatch[permutation, :, :, :]
			trainLabel = trainLabel[permutation, :]
			permutation = np.random.permutation(testPatch.shape[0])
			testPatch = testPatch[permutation, :, :, :]
			testLabel = testLabel[permutation, :]
			print("randomized")

		iter = dataloader.trainNum // BATCH_SIZE
		with tqdm(total=iter, desc="epoch %3d" % (epoch + 1)) as pbar:
			for i in range(iter):
				batch_x = trainPatch[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :, :, :]
				batch_y = trainLabel[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :]
				_, batchLoss, trainAcc = sess.run([optimizer, loss, accuracy], feed_dict={x: batch_x, y: batch_y})
				pbar.set_postfix_str("loss: %.3f, accuracy:%.2f" % (batchLoss, trainAcc * 100))
				pbar.update(1)

			if iter * BATCH_SIZE < dataloader.trainNum:
				batch_x = trainPatch[iter * BATCH_SIZE:, :, :, :]
				batch_y = trainLabel[iter * BATCH_SIZE:, :]
				_, batchLoss, trainAcc = sess.run([optimizer, loss, accuracy], feed_dict={x: batch_x, y: batch_y})

			idx = np.random.choice(dataloader.testNum, size=BATCH_SIZE, replace=False)
			# Use the random index to select random images and labels.
			test_batch_x = testPatch[idx, :, :, :]
			test_batch_y = testLabel[idx, :]
			ac, ls = sess.run([accuracy, loss], feed_dict={x: test_batch_x, y: test_batch_y})
			tqdm.write('Test Data Eval: Test Accuracy = %.4f, Test Cost =%.4f' % (ac, ls))
