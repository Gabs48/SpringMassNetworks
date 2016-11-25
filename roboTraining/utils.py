import errno
import itertools
import numpy as np
from datetime import datetime
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import neurolab as nl
import pickle
import time


""" extra functionalities needed by other modules in the roboTraining package """

CTE_POWERPOINT = False # Whether to save figures for Latex documents or powerpoints

def isCallable(obj):
	""" Returns boolean whether object can be called (like a function)"""
	return hasattr(obj, '__call__')

def findIndex(tab, name):
	"""Find an index in a tab"""

	index = [-1, -1]
	i = 0;
	find = False
	for row in tab:
		j = 0;
		for cell in row:
			if cell.find(name) != -1:
				index = [i, j]
				find = True
				break
			j += 1
		if find:
			break;
		i += 1

	return index

def connections2Array(matrix, connections):
	""" Extract the parameter values from matrix
	:parameters:
		- matrix: matrix
			symmetric, sparse matrix (eg. spring constants between the nodes)
		- connection: matrix
			symmetric matrix indicating which values are actual parameters

		An example can be found at _test_array_to_connections()
	"""
	noNodes = matrix.shape[0]
	upperTriangle = np.triu_indices(noNodes)
	matrixArray = matrix[upperTriangle][connections[upperTriangle].astype(bool)]
	return matrixArray

def connections2List(matrix, connections):
	""" IIdem as connection to array but place it in a python list"""
	liste = list(connections2Array(matrix, connections))
	return liste

def array2Connections(array, connections):
	""" Reconstruct the sparse matrix, whose parameter values reside in array

	:parameters:
		- array: array
			contains the non-zero values of A
		- connections: matrix
			indicates the non-zero elements of A
	"""

	noNodes = connections.shape[0]
	matrix = np.zeros((noNodes, noNodes))
	upperTriangle = np.triu_indices(noNodes)
	longArray = np.zeros((noNodes*(noNodes+1)/2,))
	longArray[connections[upperTriangle].astype(bool)] = array
	matrix[upperTriangle] = longArray
	matrix += matrix.T
	return matrix

def list2Connections(liste, connections):
	""" Reconstruct the sparse matrix, whose parameter values reside in a liste

	:parameters:
		- liste: list
			contains the non-zero values of A
		- connections: matrix
			indicates the non-zero elements of A
		- noNodes: int
			indicates the number of nodes in the morphology
	"""

	listLen = len(liste)
	assert listLen > 0, "the parameter list cannot be converted to a connections matrix if empty"
	noLinks = np.count_nonzero(connections != 0) / 2
	noNodes = connections.shape

	if listLen == noLinks:
		# Links property
		#print "Link reconstruction " + str(listLen)
		return array2Connections(np.array(liste), connections)

	else:
		# Homogeneous property
		# print "Homogeneous reconstruction " + str(listLen)
		return array2Connections(number2Array(liste[0], noLinks), connections)

def array2ModFactor(array, connections):
	""" Reconstruct the Modulation factor matrix from the links value"""

	matrix = array2Connections(array, connections)
	matrix[np.where(matrix == 0)] = 1
	return matrix

def number2Array(val, length):
	""" Contruct an homogeneous parameters array from a single value
	:parameters:
		- val: float
			the value to put in the array
		- length: int
			the length of the reconstructed value

		An example can be found at _test_array_to_connections()
	"""
	l = []

	for i in range(length):
		l.append(val)

	return np.array(l)

def number2List(mat):
	""" Convert a matrix with one unique number to a list of one number
	"""
	liste = []
	liste.append(float(mat[0][0]))
	return liste

def list2PosMatrix(liste):
	""" Reconstruct the position matrix, whose parameter values reside in the list

	:parameters:
		- liste: list
			contains all the value in a 1D list (row after row)
	"""

	assert len(liste) % 2 == 0, 'The length of the list should be a multiple of 2'

	array = np.array(liste)
	matrix = np.reshape(array, (2, -1))

	return matrix

def list2SquareMatrix(liste):
	""" Reconstruct the square matrix, whose parameter values reside in the list

	:parameters:
		- liste: list
			contains all the value in a 1D list (row after row)
	"""

	assert isSquare(len(liste)), 'The length of the list should be a perfect square number'

	length = int(math.sqrt(len(liste)))
	array = np.array(liste)
	matrix = np.reshape(array, (-1, length))

	return matrix

def isSquare(posInt):
	x = posInt // 2
	seen = set([x])
	while x * x != posInt:
		x = (x + (posInt // x)) // 2
		if x in seen: return False
		seen.add(x)
	return True

def num2str(num):
	""" convert a number into a short string"""
	if abs(num) < 0.01 and abs(num) > 1e-50 or abs(num) > 1E4:
		numFormat =  ".2e"
	elif abs(round(num) - num) < 0.001 or abs(num) > 1E2:
		numFormat = ".0f"
	elif abs(num) > 1E1:
		numFormat = ".1f"
	else:
		numFormat = ".2f"
	return ("{:" + numFormat + "}").format(num)

def getLength(array):
	try: 
		num = np.size(array, axis = 0)
	except:
		num = 0
	if not isinstance(array, basestring):
		return num
	else:
		return 0

def movingAverage(array, period):
	averageArray = np.zeros_like(array)
	assert period%2 == 1, 'the period must be odd'
	minPeriod = int((period-1) /2)
	maxPeriod = int((period+1) /2)
	arrayLength = len(array)
	for i in range(arrayLength):
		sum = 0;
		count = 0.0;
		minRange = max(0, i - minPeriod)
		maxRange = min(arrayLength, i + maxPeriod)
		for j in range(minRange, maxRange):
			sum+= array[j]
			count+= 1
		averageArray[i] = sum / count
	return averageArray

def getItemAt(array, index):
	try:
		result = array[index]
		assert not isinstance(array, basestring)
	except:
		assert index < getLength(array) or 0 == getLength(array), "the index is too large for the requested operation"
		result = array
	return result

def ceilToRoundNumber(number):
	base =  10 * int(np.log10(number))
	return int(np.ceil(number / base) * base)

def mkdir_p(path):
	"""Method to create a directory only if it does not already exists (tested only on linux)"""

	try:
		os.makedirs(path)
	except OSError as exc:  # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise

def timestamp():
	""" Create a timestap string """

	return time.strftime("%Y%m%d-%H%M%S", time.localtime())

def dump_pickle(data, filename):
	""" Dumps data to a pkl file """

	if not filename.endswith('.pkl'):
		raise ValueError('Pickle files should end with .pkl, but got %s instead' % path)

	with open(filename, 'wb') as pkl_file:
		pickle.dump(data, pkl_file, pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
	""" Load data from a pickle file """

	with open(filename, 'rb') as pkl_file:
		return pickle.load(pkl_file)

class Save(object):
	""" general class for saving"""
	configname = 'config'

	def __init__(self, object, name, directory="default", timestamp = True, floatFormat = '.5f'):
		self.object = object
		self.name = name
		self.directory = directory
		self.timestamp = timestamp
		self.floatFormat = floatFormat
		self.inProgress = False
		self.transpose = False

	def generateName(self, type = "", ext = "", name = None):
		if name == None:
			name = self.name

		if not self.inProgress:
			if self.timestamp:
				append ="/" + "{:%Y%m%d_%H%M%S}".format(datetime.now())
			else: append = ""

			self.folder = self.directory +"/" + name + append + "/"
			self.inProgress = True
			
			if not os.path.exists( self.folder):
				os.makedirs(self.folder)

		return self.folder + name + type + ext

	def save(self, data = None, name = None, close = True, transpose=False):
		""" write data and object to CSV file, input can be object, array like, string, dictionairy"""

		if isinstance(data, dict):
			for key in data:
				if key =="parameter":
					Save.toCSV(data[key], self.generateName(key), floatFormat="0.10f")
				else:
					Save.toCSV(data[key], self.generateName(key), self.floatFormat, transpose=transpose)
		elif not data is None:
			Save.toCSV(data, self.generateName(), self.floatFormat, transpose=transpose)

		Save.toCSV(self.object, self.generateName(Save.configname), self.floatFormat, transpose=transpose)
		if close:
			self.close()

	def close(self):
		self.inProgress = False;

	@staticmethod
	def toCSVstring(input , level = 0, floatFormat = '.5f', transpose=False):
		""" convert an object, array of objects, matrix, vector, ... into a CSV-style string"""
		beginLine = "\n" + "; " * level
		string = beginLine
		if transpose:
			string = ""
		if hasattr(input, "param"):
			string += (beginLine).join([attribute + (': ; ') + Save.toCSVstring( getattr(input, attribute),level+1, floatFormat) for attribute in input.param])
		else: 
			input = np.array(input)
			if input.ndim == 3:
				string += ("\n"+beginLine).join([Save.toCSVstring(num, 0, floatFormat) for num in input])
			elif input.ndim == 2:
				if not transpose:
					string += (beginLine).join([Save.toCSVstring(num, 0, floatFormat) for num in input])
				else:
					string += (beginLine).join([Save.toCSVstring(num, 0, floatFormat) for num in input.T])
			elif input.ndim == 1:
				if isinstance(input[0], dict):
					for kwarg in input:
						string += (beginLine).join([';' + key + ';' + Save.toCSVstring(kwarg[key], 0, floatFormat) for key in kwarg])
						string += "\n" + beginLine
				elif hasattr(input[0], "param"):
					string = '; '.join([Save.toCSVstring(object, level + 1, floatFormat) for object in input])
				else:
					string = '; '.join([('%'+floatFormat) % num for num in input])
			elif input.ndim == 0:
				if isinstance(input, float):
					 string ="{:"+floatFormat+"}".format(input)
				else:
					 string = str(input)
			else: raise (NotImplementedError), " the function toCSVstring supports but three dimensionsional arrays"
		return string

	@staticmethod
	def toCSV( content, fileName, floatFormat = '.5f', transpose=False):
		with open(fileName+".csv", "w") as csv_file:
			csv_file.write(Save.toCSVstring(content, 0,floatFormat, transpose=transpose))

class SpaceList(object):
	""" class which allows easy bundeling of x,y and eventualy z positions
		attributes:
		dim: whether Spacelist is two or threedimensional
		matrix: matrix with x,y (and z) data per node
		"""
	__array_priority__ = 1000 # assure numpy calls with SpaceList multiplication are handled by SpaceList
	param = ["matrix"]
	def __init__(self, *args):
		""" create a vector instance, no copy of data is created!!! """
		if len(args) == 0:
			self.dim = 0
		if len(args) == 1:
			assert args[0].ndim == 2, "if only one argument is used a matrix must be given"
			assert np.shape(args[0])[0] in (2, 3)
			self.dim = np.shape(args[0])[0]
			self.matrix = args[0]
		elif len(args) == 2:
			assert args[0].ndim == 1	, "each element must be an numpy array"
			assert args[1].ndim == 1	, "each element must be an numpy array"
			self.matrix = np.array( [args[0], args[1]], copy = False)
			self.dim = 2
		elif len(args) == 3:
			assert args[0].ndim == 1	, "each element must be an numpy array"
			assert args[1].ndim == 1	, "each element must be an numpy array"
			assert args[2].ndim == 1	, "each element must be an numpy array"
			self.dim = 3
			self.matrix = np.array( [args[0], args[1], args[2]], copy = False)
		else: raise(NotImplementedError)

	def __getattribute__(self, name):
		if name == 'x':
			return self.matrix[0,:]
		if name == 'y':
			return self.matrix[1,:]
		if name == 'z':
			assert self.dim ==3, " a third coordinate of a two dimensional vector is demanded"
			return self.matrix[2,:]
		if name == 'shape':
			return np.shape(self.matrix)
		else: return object.__getattribute__(self,name)

	def __setattr__(self, name, value):
		if name == 'x':
			self.matrix[0,:] = value
		elif name == 'y':
			self.matrix[1,:] = value
		elif name == 'z':
			assert self.dim ==3, " a 3rd coordinate of a 2 dimensional vector is demanded"
			self.matrix[2,:] = value
		else: super(SpaceList, self).__setattr__(name, value)

	def __add__(left, right):
		return SpaceList(left.matrix + right.matrix)

	def __iadd__(self, right):
		self.matrix = self.matrix + right.matrix
		return self

	def __mul__(self, other):
		""" multipication with SpaceList, np array or scalar """
		if hasattr(other, "matrix"):
			return SpaceList(self.matrix * other.matrix)
		return SpaceList(self.matrix * other)

	__rmul__ = __mul__ # make no distiction between left and right multiplication

	def __div__(self, other):
		# NOTE not Python 3.0 compatible 
		""" division with SpaceList, np array or scalar """
		if hasattr(other, "matrix"):
			return SpaceList(self.matrix / other.matrix)
		return SpaceList(self.matrix / other)

	def __imul__(self,other):
		try:
			self.matrix = self.matrix * other.matrix
		except AttributeError:
			self.matrix = self.matrix * other
		return self

	def copy(self):
		""" create a copy of a vector """
		return SpaceList(np.array(self.matrix))

	def __str__(self):
		string= "X: " + str(self.x) + " \n Y:" +str (self.y)
		if self.dim == 3:
			string += "\n Z: " + str(self.z)
		return string

	def get(self, nodenumber):
		return np.array(self.matrix[:,nodenumber])

	def getDistance(self, node1, node2):
		coord1 = self.get(node1)
		coord2 = self.get(node2)
		return np.sum( (coord1 - coord2) ** 2)

	def getnoNodes(self):
			return np.size(self.matrix,1)

	def getDifference(self):
		""" returns 2 or 3 matrices with the element on position (i,j) gives the difference between xi and xj """
		difx = self.x[:, np.newaxis] - self.x[np.newaxis, :]
		dify = self.y[:, np.newaxis] - self.y[np.newaxis, :]
		if self.dim == 3:
			difz = self.z[:, np.newaxis] - self.z[np.newaxis, :]
			return difx, dify, difz
		else: return difx, dify

	def getArray(self):
		""" returns an array with all the matrix values """

		return np.reshape(self.matrix, (1, -1))

	def ground(pos, speed):
		for i in xrange(pos.getnoNodes()):
			if pos.matrix[1,i] < 0: # Y component goes below the ground plane
				pos.matrix[1,i]  =0 
				speed.matrix[:,i] = 0 

class Plot(object):
	@staticmethod
	def initPlot(proj=None):
		if proj == None:
			return plt.subplots(facecolor=(1,1,1), figsize=(10, 6), dpi=150)
		elif proj == "3d":
			fig = plt.figure(facecolor=(1,1,1), figsize=(10, 6), dpi=150)
			ax = fig.add_subplot(111, projection="3d")
			return fig, ax

	@staticmethod
	def initBarPlot(labels, values, xlabel , ylabel, label = None, barWidth = 0.5):
		""" values is the array of obtained values corresponding to the labels """
		fig, ax = Plot.initPlot()
		fontsize = 20
		Plot.addbars(ax, values, '#5555ff', barWidth, label = label)

		noVal = np.size(values, axis = 0)
		indices = np.arange(noVal)

		plt.xlabel(xlabel, fontsize = fontsize)
		plt.ylabel(ylabel, fontsize = fontsize)
		plt.xticks(indices + barWidth / 2, labels)

		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.yaxis.grid() # vertical lines

		plt.tick_params(labelsize = fontsize)
		plt.tick_params(axis='x', which='both', bottom='on', top='off')
		plt.tick_params(axis='y', which='both', left='off', right='off')
		plt.tight_layout()
		return fig, ax

	@staticmethod
	def addbars(ax, values, color, barWidth, label = None, rank = 0):
		mew = 5 #Marker Edge Width
		alpha = 1
		markersize = 15
		error_config = errorConfig = {'ecolor': '0.2', 'elinewidth' : 4, 'capthick' : 4,  'capsize' : 10}

		noVal = np.size(values, axis = 0)
		indices = np.arange(noVal) + rank * barWidth

		avg = np.average(values, axis = 1)
		min = np.min(values, axis = 1)
		max = np.max(values, axis = 1)
		std = np.std(values, axis = 1) / np.sqrt(np.size(values, axis = 1))
		
		rect = ax.bar(indices, avg, barWidth, color = color, alpha = alpha, yerr= std, error_kw=error_config, label = label)

		plt.plot(indices + barWidth / 2, min,"k_", markersize = markersize, mew = mew)
		plt.plot(indices + barWidth / 2, max,"k_", markersize = markersize, mew = mew)
	
	@staticmethod
	def configurePlot(fig, ax, xLabel, yLabel, legend = True, legendLocation = 'upper center', size = 'normal'):
		fig.subplots_adjust(bottom=0.15)
		ax.tick_params(axis='x', which='both', bottom='on', top='off')
		ax.tick_params(axis='y', which='both', left='on', right='off')
		if CTE_POWERPOINT:
			labelsize = 22
		elif size == 'huge':
			labelsize = 30
		elif size == 'small':
			labelsize = 14
		elif size == 'normal':
			labelsize = 18
		else: raise NotImplementedError, 'the option ' + size + 'is not yet implemented'

		ax.tick_params(labelsize = labelsize)
		ax.set_xlabel(xLabel, size = labelsize)
		ax.set_ylabel(yLabel, size = labelsize)
		try:
			fig.tight_layout()
		except:
			print 'no tight layout'
		if legend:
			if legendLocation == 'upper center':
				bbox1 = 0.5
				bbox2 = 1
			elif legendLocation == 'upper right':
				bbox1 = 1
				bbox2 = 1
			elif legendLocation == 'lower center':
				bbox1 = 0.5
				bbox2 = 0
			elif legendLocation == 'lower right':
				bbox1 = 1
				bbox2 = 0
			elif legendLocation == 'lower left':
				bbox1 = 0
				bbox2 = 0
			ax.legend(prop={'size':labelsize}, loc = legendLocation, bbox_to_anchor = (bbox1, bbox2), ncol = 3, fancybox = True, shadow = True)

	@staticmethod
	def save2eps(fig, filename, confirm = False):
		if CTE_POWERPOINT:
			folder = r"C:\Users\benonie\SkyDrive\UGent\_PRACAM-globe\presentations\second Powerpoint\img\orig"
			path = os.path.normpath(os.path.join(folder, filename +  ".png"))
			if not confirm:
				print 'POWERPOINT MODE ENABLED'
				save = raw_input("do you want to save the image: \"" + filename +  "\"\npress y for saving\n")
				if save == "y":
					confirm = True
			if confirm:
				fig.savefig(path, format = 'png', dpi = 300)
		else:
			folder = r"C:\Users\benonie\SkyDrive\UGent\_PRACAM-globe\Report\figures\plots"
			path = os.path.normpath(os.path.join(folder, filename +  ".eps"))
			if not confirm:
				save = raw_input("do you want to save the image: \"" + filename +  "\"\npress y for saving\n")
				if save == "y":
					confirm = True
			if confirm:
				fig.savefig(path, format = 'eps', dpi = 1200)
		return confirm

	@staticmethod
	def setAxes2Dplot(margin = 0.25):
		x0, x1, y0, y1 = plt.axis()
		plt.axis((x0 - margin, x1 + margin, y0 - margin, y1 + margin))

	@staticmethod
	def setAxes3Dplot(ax, min, max):
		ax.set_xlim3d(min, max)
		ax.set_ylim3d(min, max)
		ax.set_zlim3d(min, max)

	@staticmethod
	def circles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
		import numpy as np
		import matplotlib.pyplot as plt
		from matplotlib.patches import Circle
		from matplotlib.collections import PatchCollection

		if np.isscalar(c):
			kwargs.setdefault('color', c)
			c = None
		if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
		if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
		if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
		if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))

		patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
		collection = PatchCollection(patches, **kwargs)
		if c is not None:
			collection.set_array(np.asarray(c))
			collection.set_clim(vmin, vmax)

		ax = plt.gca()
		ax.add_collection(collection)
		ax.autoscale_view()
		if c is not None:
			plt.sci(collection)
		return collection

class NeuralNetwork(object):
	outputMaxSafetyMargin = 1.5
	
	def __init__(self, maxNoEpochs = 1000, architecture = [8, 1], min = 0, max = 10, goal = 0.0002):
		self.MaxNoEpochs = maxNoEpochs
		self.architecture = architecture
		self.min = min
		self.max = max
		self.goal = goal

	def train(self, trainArray, targetArray):
		""" give input with data samples being represented as column vectors """
		inp = trainArray.T
		targ = targetArray.T
		self.outputMax = np.max(np.abs(targetArray)) * self.outputMaxSafetyMargin
		dimInput = len(trainArray)
		neuralNetwork = nl.net.newff([[self.min, self.max]] * dimInput, self.architecture)
		error = neuralNetwork.train(trainArray.T, targetArray.T / self.outputMax, epochs = self.MaxNoEpochs, show = 10, goal = self.goal) 

		trainOutput = neuralNetwork.sim(trainArray.T).T * self.outputMax 
		trainError= np.sqrt(np.mean( (trainOutput - targetArray) ** 2))
		trainRelError = trainError / ( np.sqrt(np.mean(targetArray**2) - np.mean(targetArray)**2))
		self.neuralNetwork = neuralNetwork
		return self, trainOutput, trainError, trainRelError
	
	def test(self, testArray, targetArray):
		testOutput = self.neuralNetwork.sim(testArray.T).T * self.outputMax
		testError= np.sqrt(np.mean( (testOutput - targetArray) ** 2))
		testRelError = testError / ( np.sqrt(np.mean(targetArray**2) - np.mean(targetArray)**2))
		return testOutput, testError, testRelError
	
	def value(self, input):
		matrix = self.neuralNetwork.sim(np.array([input]) * self.outputMax)
		return matrix[0]

	@staticmethod
	def example():
		x = np.array([range(1000)],float) / 300
		y = x ** 2
		self = NeuralNetwork(maxNoEpochs = 10)
		neuralNetwork, output, error, relerror = self.train(x,y)
		plt.plot(x[0,:],y[0,:], label = 'target')
		plt.plot(x[0,:],output[0,:], label = 'output')
		plt.legend()
		print self.value(np.array([1]))
		plt.show(block = True)

class LinearMap(object):
	def __init__(self):
		""

	def train(self, trainArray, targetArray):
		weights, trainOutput, trainError, trainRelError = LinearMap.linearMapping( LinearMap.addOnes(trainArray), targetArray)
		self.weights = weights
		return self, trainOutput, trainError, trainRelError

	def test(self, testArray, targetArray):
		return LinearMap.testmapping( LinearMap.addOnes(testArray), targetArray, self. weights)

	def value(self, input):
		return np.dot( self.weights, np.hstack((input, [1])) )

	@staticmethod
	def addOnes( array):
		append = np.ones((1,array.shape[1]))
		return np.concatenate((array ,append), axis = 0)

	@staticmethod
	def nonLinearDataMultiplication(array):
		newArray = np.concatenate((array ,array ** 2), axis = 0)
		return LinearMap.addOnes(newArray)
	
	@staticmethod
	def linearMapping(reservoirArray, targetArray):
		inputArrayMP = np.linalg.pinv(reservoirArray)
		weights = np.dot(targetArray,inputArrayMP)
		trainOutput = np.dot( weights, reservoirArray)
		trainError= np.sqrt(np.mean( (trainOutput - targetArray) ** 2))
		trainRelError = trainError / ( np.sqrt(np.mean(targetArray**2) - np.mean(targetArray)**2))
		return weights, trainOutput, trainError, trainRelError
	
	@staticmethod
	def testmapping(reservoirArray, targetArray, weights):
		testOutput = np.dot( weights, reservoirArray)
		testError= np.sqrt(np.mean( (testOutput - targetArray) ** 2))
		testRelError = testError / ( np.sqrt(np.mean(targetArray**2) - np.mean(targetArray)**2))
		return testOutput, testError, testRelError

	@staticmethod
	def example():
		x = np.array( [range(1000)],float) / 300
		xAppend = LinearMap.nonLinearDataMultiplication(x)
		y = x ** 2
		coeff, output, error, relerror = LinearMap.linearMapping(xAppend,y)
		plt.plot(x[0,:],y[0,:], label = 'target')
		plt.plot(x[0,:],output[0,:], label = 'output')
		plt.legend()
		print LinearMap.value(np.array([10,100]), coeff)
		plt.show(block = True)
		print error
