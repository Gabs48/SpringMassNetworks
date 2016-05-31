from roboTraining.robot import *
from roboTraining.simulate import *
from roboTraining.training import *
from roboTraining.utils import Save
from roboTraining import unitTest
import csv
import numpy as np
from multiprocessing import Pool, freeze_support
import logging


# set logging properties
logger = logging.getLogger('training')
logger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler('logoutput.log')
fileHandler.setLevel(logging.DEBUG)

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fileHandler.setFormatter(formatter)
consoleHandler.setFormatter(formatter)

logger.addHandler(fileHandler)
logger.addHandler(consoleHandler)

class TrainingEnsemble(object):
	""" bundel different trainings, to export to single csv file for easy readout"""

	def __init__(self, name, paramList):
		self.name = name
		self.param = paramList
		self.trainingList= [["result"]]
		for parameter in paramList:
			self.trainingList[0].append(parameter)

	def add(self, result, **kwargs):
		""" add training to the ensemble"""
		list = [result]
		for parameter in self.param:
			list.append(kwargs[parameter])
		self.trainingList.append(list)

	def write(self):
		""" write ensemble to csv file """
		resultFile = open(self.name + ".csv",'wb')
		wr = csv.writer(resultFile, delimiter=";")
		wr.writerows(self.trainingList)

def createSimulationRobot(noNeighbours, noNodes):
	""" create the standard trainschema, robot, and simulationEnvironment for the different tests"""
	env=HardEnvironment()
	morph=SpringMorphology(noNodes = noNodes ,spring = 100, noNeighbours = noNeighbours,environment = env)
	control=SineControl(morph)
	robot=Robot(morph,control)

	timeStep = 0.01
	plotter =Plotter(plotCycle=20,plot=False);
	simulenv=SimulationEnvironment(timeStep, simulationLength = 1000, plot =plotter)
	
	trainscheme = TrainingScheme()
	trainscheme.createTrainVariable("omega",0 ,10)
	trainscheme.createTrainVariable("phase", 0 , 2*np.pi)
	trainscheme.createTrainVariable("amplitude", 0 , 0.25)
	return trainscheme, robot, simulenv,

def randomTraining(name= 'default', noNodes = 7, noNeighbours = 3):
	""" perform a random search optimization"""
	saver = Save(None, name, 'RandomTraining') 
	train = RandomTraining(*createSimulationRobot(noNeighbours, noNodes), noInstances = 1500, saver = saver)
	param, score = train.run()
	train.save()
	return param, score

def cmaTraining(name = 'default', noNodes = 7, noNeighbours = 3):
	""" perform a cma search """
	saver = Save(None, name, 'CMATraining') 
	train = CMATraining(*createSimulationRobot(noNeighbours, noNodes), maxIter = 1500, saver = saver)
	param, score = train.run()
	train.save()
	return param, score

def geneticTraining (name = 'default', noGenerations = 30, populationSize = 50, mutationSigma = 0.5 , crossover = "type", 
				selector = "tournament", crossoverRate = 0.33, mutationRate = 0.33, scaling = "exponential", 
				databaseName = 'default', noNodes = 7, noNeighbours = 3):
	""" perform a GA optimization"""
	saver = Save(None, name, 'GeneticTraining') 
	train = GeneticTraining(*createSimulationRobot(noNeighbours, noNodes), noGenerations = noGenerations, populationSize = populationSize,
				mutationSigma = mutationSigma, crossover = crossover, selector = selector, crossoverRate = crossoverRate,
				mutationRate = mutationRate, scaling = scaling, saver = saver, databaseName = databaseName)
	param, score = train.run()
	train.save()
	return param, score

def testMutationCrossover(replicationNumber):
	" perform a replication of the first run, ensure only different replicationNumbers are used simultaneously"
	name = "mutationCrossoverRun"+str(replicationNumber)
	ensemble = TrainingEnsemble(name, [ 'mutationRate', 'mutationSigma', 'crossoverRate' ])

	# parameters over which is tested
	mutationRates = [1, 0.33, 0.1, 0.01]
	mutationSigmas = [0.2, 0.5, 0.1, 0.05, 0.01]
	crossoverRates = [1, 0.33, 0.1, 0]
	
	for mutationRate in mutationRates:
		for mutationSigma in mutationSigmas:
			for crossoverRate in crossoverRates:
				kwargs = { 'mutationRate': mutationRate, 'mutationSigma': mutationSigma, 'crossoverRate': crossoverRate}
				param, score =geneticTraining(name = name, databaseName = name, scaling = "linear", crossover = "node", selector = "roulette", **kwargs)
				ensemble.add(score, **kwargs)
	ensemble.write()

def testScalingSelectorCrossovertype(replicationNumber):
	" perform a replication of the second run"
	name = "ScalingSelectorCrossovertype"+str(replicationNumber)
	ensemble = TrainingEnsemble(name, [ 'selector' , 'crossover', 'scaling'])

	#parameters over which is tested
	crossovers = ['type', 'node', 'uniform']
	scalings = ['linear', 'exponential', 'rank' , 'sigma']
	selectors = ['roulette', 'tournament']
	
	for crossover in crossovers:
		for scaling in scalings:
			for selector in selectors:
				kwargs = { 'crossover' : crossover, 'selector' : selector, 'scaling' : scaling}
				param, score =geneticTraining(name = name, databaseName = name, **kwargs)
				ensemble.add(score, **kwargs)
	ensemble.write()
	
def testPopulation(replicationNumber):
	name = "PopulationRun"+str(replicationNumber)
	ensemble = TrainingEnsemble(name, [ 'noGenerations', 'populationSize' ])
	params = [[30, 50], [20, 75], [50, 30], [75, 20]]
	for param in params:
		kwargs = { 'noGenerations': param[0], 'populationSize': param[1]}
		optparam, score =geneticTraining(name = name, databaseName = name, **kwargs)
		ensemble.add(score, **kwargs)
	ensemble.write()

def testComparison(replicationNumber, method = "random"):
	""" perform the comparison between GA or Random Search """
	name = "Comparison" + str(replicationNumber) 
	if method == "random":		# use random search
		name = "random" + name
	elif method == "ga":			# use genetic algorithms
		name = "GA" + name
	elif method == "cma":
		name = "CMA" + name
	ensemble = TrainingEnsemble(name, ['noNodes', 'noNeighbours'])

	# different robots over which test is run
	robots = [[11, 3], [11, 5], [20, 3], [20,5], [20, 7], [30,3], [30, 5]]

	for robot in robots:
		kwargs = { 'noNodes': robot[0], 'noNeighbours': robot[1]}
		try:
			if method == "random":
				param, score = randomTraining(name = name, **kwargs)
			elif method == "ga":
				param, score = geneticTraining(name = name, databaseName = name, **kwargs)
			elif method == "cma":
				param, score = cmaTraining(name = name, **kwargs)
			ensemble.add(score, **kwargs)
		except Exception as exp:
			message = "training has failed robot had noNodes :=" + str(robot[0]) + " noNeighbours:= " + str(robot[1])
			logger.error(message, exc_info=True)
	ensemble.write()

def testComparisonGA(replicationNumber):
	return testComparison(replicationNumber, method = "ga")

def testComparisonCMA(replicationNumber):
	return testComparison(replicationNumber, method = "cma")

if __name__ == '__main__':
	# initialize multiprocessing
	freeze_support()
	pool = Pool(5)
	# add tasks to the pool

	# first run
	#pool.map(testMutationCrossover, range(3))

	# second run
	#pool.map(testScalingSelectorCrossovertype, range(3))

	# third run
	#pool.map(testPopulation, range(10))
	
	# comparison (change default parameter 'random' to select GA)"""
	replnum = 10
	#pool.map(testComparison, range(replnum))
	pool.map(testComparisonCMA, range(replnum))
	#pool.map(testComparisonGA, range(replnum))



#cmaTraining(name = 'firstCMAtest',noNodes = 7, noNeighbours = 3);
#geneticTraining(name= 'firstGAtest', noNodes = 7, noNeighbours = 3);



