import numpy as np
import scipy.io
from tqdm import tqdm


class Data:
	classes, classesIndex = [], []
	allPatch, allPatchLabel = [], []
	trainPatch, testPatch = [], []
	trainLabel, testLabel = [], []
	numEachClass = []

	def __init__(self, pathName, matName, patchSize, portion):
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
			self.classes.append([])
			self.classesIndex.append([])

		# normalize and pad
		self.data = self.data.astype(float)
		for band in range(self.bands):
			# print(np.min(self.data[:,:,band]))
			self.data[:, :, band] = (self.data[:, :, band] - np.min(self.data[:, :, band])) / \
									(np.max(self.data[:, :, band]) - np.min(self.data[:, :, band]))
		padSize = patchSize // 2
		self.data = np.pad(self.data, ((padSize, padSize), (padSize, padSize), (0, 0)), "symmetric")

		self.__slice()
		self.__prepareDataByPortion(portion)

	def __patch(self, i, j):
		widthSlice = slice(i, i + self.patchSize)
		heightSlice = slice(j, j + self.patchSize)
		return self.data[heightSlice, widthSlice, :]

	def __slice(self):
		with tqdm(total=self.height * self.width, desc="slicing HSI into patches") as pbar:
			for i in range(self.height):
				for j in range(self.width):
					tmpLabel = self.label[i, j]
					tmpPatch = self.__patch(i, j)
					self.allPatchLabel.append(tmpLabel)
					self.allPatch.append(tmpPatch)
					if tmpLabel != 0:
						self.classes[tmpLabel].append(tmpPatch)
						self.classesIndex[tmpLabel].append(i * self.height + j)
					pbar.update(1)
		# self.numEachClass.append(0)
		for i in range(self.numClasses + 1):
			self.numEachClass.append(len(self.classesIndex[i]))
		# print(np.shape(self.classes[i]))

	def __prepareDataByPortion(self, portion):
		np.random.seed(0)
		# num=0
		for i in range(self.numClasses):
			label = i + 1
			index = np.random.choice(self.numEachClass[label], int((self.numEachClass[label]) * portion + 0.5),
									 replace=False)
			# num+=len(index)
			self.trainPatch.extend(self.classes[label][j] for j in index)
			self.trainLabel.extend(label for j in range(len(index)))

	# print(self.trainLabel)

	def loadTrainData(self):
		pass

	def loadTestData(self):
		pass

	def loadAllPatch(self):
		return self.allPatch, self.allPatchLabel


if __name__ == "__main__":
	pathName = []
	pathName.append(".\\data\\Indian_pines.mat")
	pathName.append(".\\data\\Indian_pines_gt.mat")
	matName = []
	matName.append("indian_pines")
	matName.append("indian_pines_gt")
	# print([5 for i in range(10)])

	data = Data(pathName, matName, 5, 0.1)
	patch, label = data.loadAllPatch()
# print(np.shape(patch[1]))
# print(type(data.slice()))
# allData=data.loadAllData()[:,:,1]
# with open("seeData.txt", "w+") as f:
# 	for i in range(5):
# 		for j in range(5):
# 			print("%.3f " % patch[0][i][j][1], end="", file=f)
# 		print(file=f)
# print(data.loadAllData()[:,:,1],file=f)
