import numpy as np
import scipy.io
from tqdm import tqdm
import time


class Data:
	classPatches, classSpectrum, classIndex = [], [], []
	allPatch, allPatchLabel, allSpectrum = [], [], []
	trainPatch, trainLabel, trainSpectrum = [], [], []
	testPatch, testLabel, testSpectrum = [], [], []
	numEachClass = []

	def __init__(self, pathName, matName, patchSize, portionOrNum):
		# load data
		self.data = scipy.io.loadmat(pathName[0])[matName[0]]
		self.label = scipy.io.loadmat(pathName[1])[matName[1]]

		# prepare some basic propertities
		self.patchSize = patchSize
		self.numClasses = len(np.unique(self.label)) - 1
		self.height = self.data.shape[0]
		self.width = self.data.shape[1]
		self.bands = self.data.shape[2]

		for i in range(self.numClasses + 1):
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

	def __patch(self, i, j):
		widthSlice = slice(i, i + self.patchSize)
		heightSlice = slice(j, j + self.patchSize)
		return self.data[heightSlice, widthSlice, :]

	def __slice(self):
		with tqdm(total=self.height * self.width, desc="slicing HSI into patches") as pbar:
			for i in range(self.height):
				for j in range(self.width):
					tmpLabel = self.label[i, j]
					tmpSpectrum = self.data[i, j, :]
					tmpPatch = self.__patch(i, j)
					self.allPatchLabel.append(tmpLabel)
					self.allPatch.append(tmpPatch)
					self.allSpectrum.append(tmpSpectrum)
					if tmpLabel != 0:
						self.classPatches[tmpLabel].append(tmpPatch)
						self.classSpectrum[tmpLabel].append(tmpSpectrum)
						self.classIndex[tmpLabel].append(i * self.height + j)
					pbar.update(1)
		# self.numEachClass.append(0)
		for i in range(self.numClasses + 1):
			self.numEachClass.append(len(self.classIndex[i]))

	# print(np.shape(self.classes[i]))

	def __prepareDataByPortion(self, portion):
		np.random.seed(0)
		with tqdm(total=self.numClasses, desc="dividing patches") as pbar:
			for i in range(self.numClasses):
				label = i + 1

				index = np.random.choice(self.numEachClass[label], int((self.numEachClass[label]) * portion + 0.5),
										 replace=False)
				self.trainPatch.extend(self.classPatches[label][j] for j in index)
				self.trainSpectrum.extend(self.classSpectrum[label][j] for j in index)
				self.trainLabel.extend(label for j in range(len(index)))

				index = np.setdiff1d(range(self.numEachClass[label]), index)
				self.testLabel.extend(label for j in range(len(index)))
				self.testPatch.extend(self.classPatches[label][j] for j in index)
				self.testSpectrum.extend(self.classSpectrum[label][j] for j in index)
				pbar.update(1)

		# time.sleep(0.1)
		print(np.shape(self.trainSpectrum), np.shape(self.trainPatch), np.shape(self.testSpectrum),
			  np.shape(self.testPatch))

	def __prepareDataByNum(self, num):
		np.random.seed(0)
		with tqdm(total=self.numClasses, desc="dividing patches") as pbar:
			for i in range(self.numClasses):
				label = i + 1
				index = np.random.choice(self.numEachClass[label], num, replace=False)
				self.trainPatch.extend(self.classPatches[label][j] for j in index)
				self.trainSpectrum.extend(self.classSpectrum[label][j] for j in index)
				self.trainLabel.extend(label for j in range(len(index)))

				index = np.setdiff1d(range(self.numEachClass[label]), index)
				self.testLabel.extend(label for j in range(len(index)))
				self.testPatch.extend(self.classPatches[label][j] for j in index)
				self.testSpectrum.extend(self.classSpectrum[label][j] for j in index)
				pbar.update(1)

		# time.sleep(0.1)
		print(np.shape(self.trainSpectrum), np.shape(self.trainPatch),
			  np.shape(self.testSpectrum), np.shape(self.testPatch))

	def loadTrainData(self):
		return self.trainPatch, self.trainSpectrum, self.trainLabel

	def loadTestData(self):
		return self.testPatch, self.testSpectrum, self.testLabel

	def loadAllPatch(self):
		return self.allPatch, self.allSpectrum, self.allPatchLabel


if __name__ == "__main__":
	pathName = []
	pathName.append(".\\data\\Indian_pines.mat")
	pathName.append(".\\data\\Indian_pines_gt.mat")
	matName = []
	matName.append("indian_pines")
	matName.append("indian_pines_gt")
	# print([5 for i in range(10)])

	data = Data(pathName, matName, 5, 0.1)
	patch, spectrum, label = data.loadTrainData()
	print(np.shape(patch),np.shape(spectrum),np.shape(label))
# print(np.shape(patch[1]))
# print(type(data.slice()))
# allData=data.loadAllData()[:,:,1]
# with open("seeData.txt", "w+") as f:
# 	for i in range(5):
# 		for j in range(5):
# 			print("%.3f " % patch[0][i][j][1], end="", file=f)
# 		print(file=f)
# print(data.loadAllData()[:,:,1],file=f)
