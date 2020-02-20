import numpy as np
import scipy.io
from tqdm import tqdm
from utils import convertToOneHot
import time
import random
import scipy.ndimage


class DataLoader:
	classPatches, classSpectrum, classIndex = [], [], []
	allPatch, allPatchLabel, allSpectrum = [], [], []
	trainPatch, trainLabel, trainSpectrum = [], [], []
	testPatch, testLabel, testSpectrum = [], [], []
	numEachClass = []
	trainNum = 0
	testNum = 0

	def __init__(self, pathName, matName, patchSize, portionOrNum,ratio):
		# load data
		self.data = scipy.io.loadmat(pathName[0])[matName[0]]
		self.label = scipy.io.loadmat(pathName[1])[matName[1]]

		# prepare some basic propertities
		self.patchSize = patchSize
		self.numClasses = len(np.unique(self.label)) - 1
		self.height = self.data.shape[0]
		self.width = self.data.shape[1]
		self.bands = self.data.shape[2]

		for i in range(self.numClasses):
			self.classPatches.append([])
			self.classSpectrum.append([])
			self.classIndex.append([])

		# normalize and pad
		self.data = self.data.astype(float)
		for band in range(self.bands):
			# print(np.min(self.data[:,:,band]))
			self.data[:, :, band] = (self.data[:, :, band] - np.min(self.data[:, :, band])) / \
									(np.max(self.data[:, :, band]) - np.min(self.data[:, :, band]))
		padSize = patchSize // 2
		self.data = np.pad(self.data, ((padSize, padSize), (padSize, padSize), (0, 0)), "symmetric")

		self.__slice()
		if portionOrNum < 1:
			self.__prepareDataByPortion(portionOrNum)
		else:
			self.__prepareDataByNum(portionOrNum)
		if ratio!=0:
			self.__dataAugment(ratio)

		self.trainLabel = np.array(self.trainLabel)
		self.trainPatch = np.array(self.trainPatch)
		self.trainSpectrum = np.array(self.trainSpectrum)
		self.testLabel = np.array(self.testLabel)
		self.testPatch = np.array(self.testPatch)
		self.testSpectrum = np.array(self.testSpectrum)

		self.trainLabel = convertToOneHot(self.trainLabel, num_classes=self.numClasses)
		self.testLabel = convertToOneHot(self.testLabel, num_classes=self.numClasses)

	def __patch(self, i, j):
		widthSlice = slice(i, i + self.patchSize)
		heightSlice = slice(j, j + self.patchSize)
		return self.data[heightSlice, widthSlice, :]

	def __slice(self):
		with tqdm(total=self.height * self.width, desc="slicing ") as pbar:
			for i in range(self.height):
				for j in range(self.width):
					tmpLabel = self.label[i, j]
					tmpSpectrum = self.data[i, j, :]
					tmpPatch = self.__patch(i, j)
					self.allPatchLabel.append(tmpLabel)
					self.allPatch.append(tmpPatch)
					self.allSpectrum.append(tmpSpectrum)
					if tmpLabel != 0:
						self.classPatches[tmpLabel - 1].append(tmpPatch)
						self.classSpectrum[tmpLabel - 1].append(tmpSpectrum)
						self.classIndex[tmpLabel - 1].append(i * self.height + j)
					pbar.update(1)
		# self.numEachClass.append(0)
		for i in range(self.numClasses):
			self.numEachClass.append(len(self.classIndex[i]))

	def __prepareDataByPortion(self, portion):
		np.random.seed(0)
		with tqdm(total=self.numClasses, desc="dividing") as pbar:
			for i in range(self.numClasses):
				label = i
				index = np.random.choice(self.numEachClass[label], int((self.numEachClass[label]) * portion + 0.5),
										 replace=False)
				self.trainPatch.extend(self.classPatches[label][j] for j in index)
				self.trainSpectrum.extend(self.classSpectrum[label][j] for j in index)
				self.trainLabel.extend(label for j in range(len(index)))
				self.trainNum += len(index)

				index = np.setdiff1d(range(self.numEachClass[label]), index)
				self.testLabel.extend(label for j in range(len(index)))
				self.testPatch.extend(self.classPatches[label][j] for j in index)
				self.testSpectrum.extend(self.classSpectrum[label][j] for j in index)
				self.testNum += len(index)

				pbar.update(1)

	def __prepareDataByNum(self, num):
		np.random.seed(0)
		with tqdm(total=self.numClasses, desc="dividing patches") as pbar:
			for i in range(self.numClasses):
				label = i
				index = np.random.choice(self.numEachClass[label], num, replace=False)
				self.trainPatch.extend(self.classPatches[label][j] for j in index)
				self.trainSpectrum.extend(self.classSpectrum[label][j] for j in index)
				self.trainLabel.extend(label for j in range(len(index)))
				self.trainNum += len(index)

				index = np.setdiff1d(range(self.numEachClass[label]), index)
				self.testLabel.extend(label for j in range(len(index)))
				self.testPatch.extend(self.classPatches[label][j] for j in index)
				self.testSpectrum.extend(self.classSpectrum[label][j] for j in index)
				self.testNum += len(index)

				pbar.update(1)

	def __dataAugment(self, ratio):
		index = np.random.choice(range(self.trainNum), int(self.trainNum*ratio), replace=False)
		udPatch, udLabel, udSpectrum = [], [], []
		lrPatch, lrLabel, lrSpectrum = [], [], []
		noisePatch, noiseLabel, noiseSpectrum = [], [], []
		angelPatch, angelLabel, angelSpectrum = [], [], []
		with tqdm(total=len(index),desc="augmenting") as pbar:
			for i in index:
				udPatch.append(np.flipud(self.trainPatch[i]))
				udSpectrum.append(self.trainSpectrum[i])
				udLabel.append(self.trainLabel[i])

				lrPatch.append(np.fliplr(self.trainPatch[i]))
				lrSpectrum.append(self.trainSpectrum[i])
				lrLabel.append(self.trainLabel[i])

				noisePatch.append(self.trainPatch[i] + np.random.normal(0, 0.01, size=np.shape(self.trainPatch[0])))
				noiseSpectrum.append(self.trainSpectrum[i])
				noiseLabel.append(self.trainLabel[i])

				angel = random.randrange(-180, 180, 30)
				angelPatch.append(scipy.ndimage.interpolation.rotate(self.trainPatch[i], angel, axes=(1, 0),
																	 reshape=False, output=None, order=3,
																	 mode='constant', cval=0.0, prefilter=False))
				angelSpectrum.append(self.trainSpectrum[i])
				angelLabel.append(self.trainLabel[i])

				pbar.update(1)
		# print(np.shape(self.trainPatch),type(self.trainPatch))
		self.trainPatch.extend(udPatch[i] for i in range(len(index)))
		self.trainSpectrum.extend(udSpectrum[i] for i in range(len(index)))
		self.trainLabel.extend(udLabel[i] for i in range(len(index)))

		self.trainPatch.extend(lrPatch[i] for i in range(len(index)))
		self.trainSpectrum.extend(lrSpectrum[i] for i in range(len(index)))
		self.trainLabel.extend(lrLabel[i] for i in range(len(index)))

		# print(np.shape(noisePatch),type(noisePatch))
		self.trainPatch.extend(noisePatch[i] for i in range(len(index)))
		self.trainSpectrum.extend(noiseSpectrum[i] for i in range(len(index)))
		self.trainLabel.extend(noiseLabel[i] for i in range(len(index)))

		# print(np.shape(angelPatch), type(angelPatch))
		self.trainPatch.extend(angelPatch[i] for i in range(len(index)))
		self.trainSpectrum.extend(angelSpectrum[i] for i in range(len(index)))
		self.trainLabel.extend(angelLabel[i] for i in range(len(index)))

	# time.sleep(0.1)
	# print(np.shape(self.trainSpectrum), np.shape(self.trainPatch),
	# 	  np.shape(self.testSpectrum), np.shape(self.testPatch))

	def loadTrainData(self):
		return self.trainPatch, self.trainSpectrum, self.trainLabel

	def loadTestData(self):
		return self.testPatch, self.testSpectrum, self.testLabel

	def loadAllPatch(self):
		return self.allPatch, self.allSpectrum, self.allPatchLabel

	def loadTrainPatchOnly(self):
		return self.trainPatch, self.trainLabel

	def loadTestPatchOnly(self):
		return self.testPatch, self.testLabel

	def loadAllPatchOnly(self):
		return self.allPatch, self.allPatchLabel


if __name__ == "__main__":
	with tqdm(total=100, desc="processing") as pbar:
		for i in range(100):
			pbar.set_postfix_str("lost: %3d, left: %3d" % (i, 100 - i))
			pbar.update(1)
			time.sleep(0.1)
	exit(0)

	pathName = []
	pathName.append(".\\data\\Indian_pines.mat")
	pathName.append(".\\data\\Indian_pines_gt.mat")
	matName = []
	matName.append("indian_pines")
	matName.append("indian_pines_gt")
	# print([5 for i in range(10)])

	data = DataLoader(pathName, matName, 5, 0.1)
	patch, label = data.loadTrainPatchOnly()
# print(np.shape(patch), np.shape(label))
