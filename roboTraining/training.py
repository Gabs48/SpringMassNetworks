import abc
import csv
import os.path
import sqlite3
import numpy as np
import copy
from utils import num2str, Plot
from types import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
import platform
import time

from robot import Robot 
from simulate import Simulation, VerletSimulation
from pyevolve import (G2DList,GSimpleGA, Initializators, DBAdapters,
					 Mutators, Selectors, Crossovers, Consts, Scaling)
import cma

class TrainingVariable(object):
	""" A class for the handling of variables that will be used in the optimization of the robot"""
	param= ['name', 'min', 'max']
	def __init__(self,name,min,max):
		self.name=name
		self.min = min
		self.max = max

	def normalize(self, value):
		"""normalize value or list/array of values so it always lies between 0 and 1"""
		if type(value) is ListType:
			value=np.array(value)
		return 1.0*(value - self.min)/(self.max-self.min)

	def denormalize(self,value):
		""" calculate the original (non-normalized) value of a normalized value or list/array of values """
		if type(value) is ListType:
			value=np.array(value)
		return value*(self.max-self.min)+self.min

class TrainingScheme(object):
	""" collection of TrainVariables and associated methods"""
	param = ["trainableParams"]
	def __init__(self):
		self.trainableParams= []
		self.names=[]
	
	def addTrainVariable(self,trainVariable):
		""" add a variable to be trained to the TrainingScheme"""
		self.trainableParams.append(trainVariable)
		self.names.append(trainVariable.name)

	def createTrainVariable(self,name,min,max):
		""" create and add a variable that should be trained to the TraingScheme"""
		self.addTrainVariable(TrainingVariable(name,min,max))

	def robot2normalizedMatrix(self,robot):
		""" --- convert the requested robot parameters to a matrix ---
		
		-- parameters --
		- robot : Robot
			robot that should be converted

		-- output --
		- matrix : matrix
			normalized matrix of which the headers are given by the names attribute
		"""
		matrix = robot.robot2matrix(self.names);
		for i in range(len(self.trainableParams)):
			matrix[i,:]=self.trainableParams[i].normalize(matrix[i,:])
		return matrix

	def loadCSV(self, fileName, index, n_param=3):
		"""Load the training parameters from a config file and its index"""

		with open(fileName, 'r') as csvfile:
			tab = list(csv.reader(csvfile, delimiter=';', quotechar='|'))
			
			ii = 0
			mult = n_param + 2
			for i in range(mult * index + 2, mult * index + mult) :
				if ii == 0:
					matrix = np.array([float(it) for it in tab[i]])
				if ii == 1:
					matrix = np.vstack((matrix, np.array([float(it) for it in tab[i]])))
				if ii == 2:
					matrix = np.vstack((matrix, np.array([float(it) for it in tab[i]])))
				ii += 1

		return matrix

	def normalizedMatrix2robot(self,matrix,robot):
		""" --- update the robot to a matrix of normalized parameters together with a list of parameters  ---
		
		-- parameters --
		- matrix : matrix
			matrix of which the headers are given by paramlist
		- robot : Robot
			the robot on which the parameters should be updated

		-- parameters --
		- robot : Robot
			robot with parameters updated according to the matrix
		"""
		assert matrix.ndim == 2, "input matrix with parameters should be two dimensional"
		assert np.all (matrix <= 1), "matrix values cannot be larger than one"
		assert np.all (matrix >= 0), "matrix values cannot be smaller than zero"

		matrix = np.array(matrix)
		for i in range( len( self.trainableParams)):
			matrix[i, :] = self.trainableParams[ i].denormalize( matrix[i, :])
		return robot.matrix2robot(self.names,matrix,True)

class PartialTrainingScheme(TrainingScheme):
	def __init__(self, trainNodes = 0, noConnections = 0, robot = None, fraction = -1):
		if robot != None:
			noConnections = robot.getNoConnections()
		if fraction >= 0:
			trainNodes = int(noConnections * fraction)
		self.noConnections = noConnections
		if np.shape(trainNodes) == ():
			random = np.random.permutation(np.arange(noConnections))
			self.trainNodes = random[0:trainNodes]
		else: self.trainNodes = trainNodes
		self.default = np.zeros((0,noConnections))
		super(PartialTrainingScheme, self).__init__()

	def addTrainVariable(self,trainVariable, zero = False):
		""" add a variable to be trained to the TrainingScheme"""
		if zero:
			append = np.zeros((1, self.noConnections))
		else:
			append = np.random.rand(1, self.noConnections)
		self.default = np.vstack((self.default, append))
		# append DEFAULT
		super(PartialTrainingScheme, self).addTrainVariable(trainVariable)

	def createTrainVariable(self,name,min,max, zero = False):
		""" create and add a variable that should be trained to the TraingScheme"""
		self.addTrainVariable(TrainingVariable(name,min,max), zero = zero)

	def robot2normalizedMatrix(self,robot):
		partialMatrix = self.createPartialMatrix(super(PartialTrainingScheme, self).robot2normalizedMatrix(robot))
		return partialMatrix

	def normalizedMatrix2robot(self, matrix , robot):
		fullMatrix = self.createFullMatrix(matrix)
		return super(PartialTrainingScheme, self).normalizedMatrix2robot(fullMatrix, robot)

	def createFullMatrix(self, matrix):
		fullMatrix = np.copy(self.default)
		fullMatrix[:,self.trainNodes] = matrix
		return fullMatrix

	def createPartialMatrix(self, matrix):
		return matrix[:, self.trainNodes]

class Training(object):
	__metaclass__ = abc.ABCMeta
	""" Abstract Class for training a robot
	
		---methods for member classes ---
		- run : generates parameterlists and calls evaluateParam( ) """

	# define parameters to be saved
	param = ['optimalscore', 'maximization','trainscheme','simulEnv', 'robot']

	def __init__( self, trainscheme, robot,simulEnv, saver, showIntermediateResults, keepIntermediateParameters, maximization):
		self.robot = robot
		self.trainscheme = trainscheme
		self.simulEnv = simulEnv
		self.shapeParamList = np.shape(trainscheme.robot2normalizedMatrix(robot))
		self.showIntermediateResults = showIntermediateResults
		self.keepIntermediateParameters = keepIntermediateParameters
		self.bestParameters = None;
		if maximization:
			self.optimalscore = float("-inf");
		else:
			self.optimalscore = float("inf");
		self.scoreHistory = []
		self.parameterHistory = []
		self.maximization = maximization
		self.saver = saver
		
	def addResult(self,parameterslist, score):
		""" for each simulation append relevant data (scores, parameters) to lists"""
		# result optimal for either maximization or minimization
		if self.maximization == (score > self.optimalscore):
			self.bestParameters = parameterslist
			self.optimalscore = score
		self.scoreHistory.append(score)
		
		if self.keepIntermediateParameters:
			self.parameterHistory.append(parameterslist)

 		if self.showIntermediateResults:
			# Get machine name
			comm = MPI.COMM_WORLD
			rank = comm.Get_rank()
			size = comm.Get_size()
			machine = platform.node()
			print("-- " + machine + " (" + str(rank+1) + "/" + str(size) + ")" + " -- iteration " + \
				num2str(len(self.scoreHistory)) + "\t has score \t" + num2str(score) )

	def addResults(self, ScoreData):
		""" add list of results """
		assert isinstance(ScoreData, list)
		for scores in ScoreData:
			self.scoreHistory+= scores

	def evaluateParam(self, param):
		""" calculate the performance index of a parameter set"""
		self.trainscheme.normalizedMatrix2robot(param, self.robot)
		# FIX NOISE
		if self.simulEnv.verlet:
			score = VerletSimulation(self.simulEnv, self.robot).runSimulation()
		else:
			score = Simulation(self.simulEnv, self.robot).runSimulation()
		self.addResult(param, score)
		return score
	
	def save(self, name = None, savePlot = True): 
		""" save the training with all relevant data and plots to different files"""
		if self.saver is not None:
			self.saver.object = self
			data = {"score" : self.scoreHistory}
			if self.keepIntermediateParameters:
				data["parameter"] = self.parameterHistory

			# allow childclasses to add extra data
			self.addData(data)

			self.saver.save(data, name, close = False)
			if savePlot:
				self.plot(show = False)
				plt.savefig(self.saver.generateName('plot','.eps'), format = 'eps', dpi = 1200)
				plt.close()
			self.saver.close()

	def addData(self, data):
		""" custom method to add extra data to the save dictionary"""

	def plot(self , title = 'Training Scores', show = True):
		""" generate a plot of the training"""
		fig, ax = Plot.initPlot()
		for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
			ax.get_xticklabels() + ax.get_yticklabels()):
			item.set_fontsize(17)
		self.generatePlot()
		plt.title(title + "     optimum  = " + num2str(self.optimalscore) ) # EXTEND automatic names
		Plot.configurePlot(fig, ax, 'Temp', 'Temp', legend = True, legendLocation = 'lower center')
		plt.xlabel(self.xlabel)
		plt.ylabel('Distance Traveled') # TODO generalize
		if show: plt.show()

	def generatePlot(self):
		""" plot data to plot (can be overwritten in child class) """
		barWidth = 0.35
		opacity = 0.4
		xlabel = np.arange( len(self.scoreHistory))
		plt.bar(xlabel, self.scoreHistory, barWidth, alpha = opacity, color = 'b', label = 'distance traveled')

	@ abc.abstractmethod
	def run(self):
		""" run the training algorithm """

class RandomTraining(Training):
	""" optimize parameter lists using uniform random generator search"""
	param = ['noInstances'] + Training.param
	xlabel = "Instance"

	def __init__( self, trainscheme , robot, simulEnv, saver = None, showIntermediateResults = True, saveAllstates = True, 
					maximization = True, noInstances= 100):
		self.noInstances = noInstances
		super(RandomTraining,self).__init__( trainscheme , robot, simulEnv, saver, showIntermediateResults, saveAllstates, maximization)

	def run(self):
		""" run the  training algorithm"""

		t_init = time.time()
		for i in range( self.noInstances):
			param = np.random.rand(*self.shapeParamList)
			self.evaluateParam(param)

		t_tot = time.time() - t_init
		return self.bestParameters, self.optimalscore, t_tot

class GeneticTraining(Training):
	"""" optimize parameters lists using Genetic Algorithms"""

	# define parameters to be saved
	param = ['mutationRate', 'mutationSigma', 'crossover', 'selector', 'crossoverRate','populationSize', 'noGenerations', 'scaling'] + Training.param
	xlabel = "Generation"

	def __init__( self, trainscheme , robot, simulEnv, saver = None, showIntermediateResults = True, saveAllstates = True,
			maximization = True, mutationSigma = 0.5 , crossover = "type", selector = "tournament", crossoverRate = 0.33,
		mutationRate = 0.33, populationSize = 30, noGenerations = 50, scaling = "exponential", databaseName = "default"):

		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()
		self.databaseName = databaseName + "_" + str(rank) + '.db'
		self.csvName = databaseName + "_" + str(rank) + '.csv'
		self.mutationSigma = mutationSigma
		self.crossover = crossover
		self.selector = selector

		self.crossoverRate = crossoverRate
		self.mutationRate = mutationRate

		self.populationSize = populationSize
		self.noGenerations= noGenerations
		self.scaling = scaling
		
		super(GeneticTraining,self).__init__( trainscheme , robot, simulEnv, saver , showIntermediateResults, saveAllstates, maximization)

	def evaluateParam(self, param):
		return super(GeneticTraining,self).evaluateParam(np.matrix(param.genomeList))
	
	def run(self):
		""" run the Genetic Training"""
		genome = G2DList.G2DList(*self.shapeParamList)
		genome.setParams(rangemin=0, rangemax=1, gauss_mu=0, gauss_sigma=self.mutationSigma)
		genome.evaluator.set(self.evaluateParam)
	
		genome.initializator.set(Initializators.G2DListInitializatorReal) 
		genome.mutator.set(Mutators.G2DListMutatorRealGaussian) 

		if self.crossover == "uniform":  # Uniform means not regarding the location
			genome.crossover.set(Crossovers.G2DListCrossoverUniform)
		elif self.crossover == "type": # keep the same type of arguments together
			genome.crossover.set(Crossovers.G2DListCrossoverSingleHPoint)
		elif self.crossover == "node": # keep parameter of the same node together
			genome.crossover.set(Crossovers.G2DListCrossoverSingleVPoint)
		else: raise NotImplementedError
	
		ga = GSimpleGA.GSimpleGA(genome)
		if self.maximization:
			ga.setMinimax(Consts.minimaxType["maximize"])
		else:
			ga.setMinimax(Consts.minimaxType["minimize"])
	
		if self.selector == "tournament":
			ga.selector.set(Selectors.GTournamentSelector)
		elif self.selector == "roulette":
			ga.selector.set(Selectors.GRouletteWheel)
		else: raise NotImplementedError
	
		ga.setCrossoverRate(self.crossoverRate)
		ga.setMutationRate(self.mutationRate)
	
		ga.setPopulationSize(self.populationSize)
		ga.setGenerations(self.noGenerations)
		
		sqlite_adapter = DBAdapters.DBSQLite(dbname=self.databaseName, identify="default") # save statistics
		csv_adapter = DBAdapters.DBFileCSV(filename=self.csvName, identify="default") # save statistics
		ga.setDBAdapter(sqlite_adapter)
		#ga.setDBAdapter(csv_adapter)
	
		pop = ga.getPopulation()
	
		if self.scaling == "sigma":
			pop.scaleMethod.set(Scaling.SigmaTruncScaling)
		elif self.scaling == "exponential":
			pop.scaleMethod.set(Scaling.ExponentialScaling)
		elif self.scaling == "rank":
			pop.scaleMethod.set(GeneticTraining.rankScaling) # change Class
		elif self.scaling == "linear":
			pop.scaleMethod.set(GeneticTraining.linearScaling) # change Class
		else:
			raise NotImplementedError

		ga.setElitism(True)
		ga.setElitismReplacement(2)

		t_init = time.time()
		ga.evolve()
		t_tot = time.time() - t_init

		best = ga.bestIndividual()
		self.data = self.fetchData()
		#self.data = self.fetchCSVData()
		return  best.genomeList, best.getRawScore(), t_tot

	def generatePlot(self, p = 0.2):
		""" generate Plot of the Genetic Training results """
		generation = np.arange(len(self.data))
		avgScore = []
		minScore = []
		maxScore = []
		lowerScore = []
		higherScore = []
		for scores in self.data:
			avgScore.append( np.mean(scores))
			minScore.append( np.min(scores))
			maxScore.append( np.max(scores))
			lowerScore.append( np.percentile(scores, 100 * p))
			higherScore.append( np.percentile(scores, 100 - 100 * p))
		
		plt.plot(generation, maxScore ,"bo", label = "maximum")
		plt.plot(generation, higherScore, "b--", label = num2str(100 * (1-p)) + "th percentile")
		plt.plot(generation, avgScore, "k", label = "average score")
		plt.plot(generation, lowerScore , "r--", label = num2str(100 * (p)) + "th percentile")
		plt.plot(generation, minScore, "ro", label = "minimum")

		plt.legend(loc = 4)

	def addData(self, data):
		""" adds extra data to be saved """
		data["data"] = self.data

	def fetchData(self, identify = "default"):
		""" retrieve relevant data from the database created by Pyevolve when running the GA """

		dbfile = self.databaseName
		if not os.path.exists(dbfile):
			print "Database file '%s' not found !" % (dbfile,)
			exit()
		conn = sqlite3.connect(dbfile)
		conn.row_factory = sqlite3.Row
		c = conn.cursor()
		
		ret = c.execute("select distinct generation from population where identify = ?", (identify,))
		generations = ret.fetchall()
		if len(generations) <= 0:
			print "No generation data found for the identify '%s' !" % (identify,)
			exit()
	
		data = [] # array in which all values will be stored
	
		for gen in generations:
			pop_tmp = [] # array with population parameters
			ret = c.execute("""select *  from population where identify = ? and generation = ? """, (identify, gen[0])) # database query
			ret_fetch = ret.fetchall()
			for it in ret_fetch:
				pop_tmp.append(it["raw"])
			data.append(pop_tmp)

		ret.close()
		conn.close()
		return data

	def fetchCSVData(self, identify = "default"):
		"""Return relevant data from the csv database created by pyevolve when running the GA"""

		dbfile = self.csvName
		if not os.path.exists(dbfile):
			print "CSV file '%s' not found !" % (dbfile,)
			exit()

		data = [] # array in which all values will be stored

		with open(dbfile, 'r') as csvfile:
			tab = list(csv.reader(csvfile, delimiter=';', quotechar='|'))

			for i, row in enumerate(tab):
				n_pop = int(row[2])
				data.extend(map(float, row[3:3+n_pop]))
		return data


	@staticmethod
	def rankScaling(pop):
		"""implememt rank scaling for the GA """
		def getScore(individual):
			return individual.score
		popCopy = copy.copy(pop.internalPop)
		popCopy.sort(key = lambda x: x.score)

		for i in xrange(len(pop)):
			pop[i].fitness = popCopy.index(pop[i])

	@staticmethod
	def linearScaling(pop):
		""" implement linear scaling for the GA """
		popMin = min(pop, key = lambda x:x.score).score
		for i in xrange(len(pop)):
			pop[i].fitness = (pop[i].score - popMin)

class CMATraining(Training):
	""" optimize parameter lists using the cma algorithm """

	param = ['maxIter','initMean','initStd'] + Training.param
	xlabel = "Instance"

	def __init__( self, trainscheme , robot, simulEnv,  saver = None, showIntermediateResults = True, saveAllstates = True, 
					maximization = True, maxIter= 100, initMean = 0.5, initStd = 0.2):
		super(CMATraining,self).__init__( trainscheme , robot, simulEnv, saver, showIntermediateResults, saveAllstates, maximization)
		self.initStd = initStd
		self.initMean = initMean
		self.maxIter = maxIter
		self.dim = self.shapeParamList[0] * self.shapeParamList[1]
		self.sigmaList = []

	def run(self):
		""" run the  training algorithm"""

		es = cma.CMAEvolutionStrategy(self.dim * [self.initMean], self.initStd, 
		{'boundary_handling': 'BoundTransform ','bounds': [0,1], 
		'maxfevals' : self.maxIter,'verbose' :-9})
		self.popSize = es.popsize
		
		t_init = time.time()
		
		while not es.stop():
			self.sigmaList.append(es.sigma)
			solutions = es.ask()
			es.tell(solutions, [self.evaluateParam(list) for list in solutions])
		self.sigmaList.append(es.sigma)
		res = es.result();

		t_tot = time.time() - t_init
		
		self.bestParameters = self.listToArray(res[0])
		self.optimalscore = self.resultTransform(res[1])
		
		return self.bestParameters, self.optimalscore, t_tot

	def evaluateParam(self, list):
		score = super(CMATraining,self).evaluateParam( self.listToArray(list))
		return self.resultTransform(score)

	def resultTransform(self, result):
		# cma does not support maximization thus we work with the negative of the objective function
		if self.maximization :
			return - result;
		else: return result;

	def arrayToList(self, array):
		temp = np.reshape(array,(1, self.dim))
		array.tolist()[0]
		return array

	def listToArray(self, list):
		temp = np.array(list);
		array = np.reshape(temp, self.shapeParamList)
		return array

	def cmaPlot(self, filename):
		cma.plot();
		cma.savefig(filename)
		cma.closefig()
