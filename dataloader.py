import numpy as np
import scipy.io
import os


class Data:
	classes, classesLabel = [], []
	allPath, allPatchLabel = [], []
	patchSize = 0

	def __init__(self, pathName, matName, patchSize):
		self.patchSize = patchSize
		self.data = scipy.io.loadmat(pathName[0])[matName[0]]
		self.label = scipy.io.loadmat(pathName[1])[matName[1]]
		self.numClasses = len(np.unique(self.label)) - 1
		self.height = self.data.shape[0]
		self.width = self.data.shape[1]
		self.bands = self.data.shape[2]

		for i in range(self.numClasses):
			self.classes.append([])
			self.classesLabel.append([])

		self.data = self.data.astype(float)
		for band in range(self.bands):
			# print(np.min(self.data[:,:,band]))
			self.data[:, :, band] = (self.data[:, :, band] - np.min(self.data[:, :, band])) \
									/ (np.max(self.data[:, :, band]) - np.min(self.data[:, :, band]))
		padSize = patchSize // 2
		self.data = np.pad(self.data, ((padSize, padSize), (padSize, padSize), (0, 0)), "symmetric")
		self.slice()

	def patch(self, i, j):
		widthSlice = slice(i, i + self.patchSize)
		heightSlice = slice(j, j + self.patchSize)
		return self.data[heightSlice, widthSlice, :]

	def slice(self):
		for i in range(self.height):
			for j in range(self.width):
				tmpLabel=self.label[i,j]
				tmpPatch=self.patch(i,j)
				self.allPatchLabel=tmpLabel
				self.allPatch=tmpPatch

	def loadTrainData(self):
		pass

	def loadTestData(self):
		pass

	def loadAllPatch(self):
		return self.allPatch,self.allPatchLabel


if __name__ == "__main__":
	pathName = []
	pathName.append(".\\data\\Indian_pines.mat")
	pathName.append(".\\data\\Indian_pines_gt.mat")
	matName = []
	matName.append("indian_pines")
	matName.append("indian_pines_gt")

	data = Data(pathName, matName, 5)
	# print(type(data.slice()))
# allData=data.loadAllData()[:,:,1]
# with open("seeData.txt","w+") as f:
# 	for i in allData:
# 		for j in i:
# 			print(j,end=" ",file=f)
# 		print(file=f)
# print(data.loadAllData()[:,:,1],file=f)
