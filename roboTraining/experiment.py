
from robot import *
from simulate import *
from training import *
from utils import *

# import jsonpickle
import math
from multiprocessing import *
from mpi4py import MPI
import numpy as np
import platform
import sys
import datetime


class Experiment(object):
	"""Class to perform standard experiments"""

	def __init__(self, fileName_="CMA", folderName_="Data", noNodes_=20, spring_=100, noNeighbours_=3, plot_=False, \
		simTimeStep_=0.005, simTime_=20, perfMetr_="powereff", controlPlot_=False, maxIter_=10000, omega_=2*np.pi*2, \
		optMethod_="CMA", maxAmplitude_=0.25, popSize_=30, mass_=1, refPower_=3600, refDist_=100, maxSpring_=200, \
		noisy_=False, trainOmega_=True):
		"""Initialize the variables lists"""

		self.fileName = fileName_
		self.folderName = folderName_
		self.noNodes = noNodes_
		self.spring = spring_
		self.mass = mass_
		self.noNeighbours = noNeighbours_
		self.plot = plot_
		self.simTimeStep = simTimeStep_
		self.simTime = simTime_
		self.perfMetr = perfMetr_
		self.controlPlot = controlPlot_
		self.maxIter = maxIter_
		self.omega = omega_
		self.optMethod = optMethod_
		self.maxAmplitude = maxAmplitude_
		self.popSize = popSize_
		self.noGen = int(self.maxIter / self.popSize)
		self.refDist = refDist_
		self.refPower = refPower_
		self.maxSpring = maxSpring_
		self.noisy = noisy_
		self.trainOmega = trainOmega_

	def run(self):
		"""Run the experiment"""

		# Init (environment, morphology, controller, simulator)
		env = HardEnvironment()
		morph = SpringMorphology(mass=self.mass, noNodes=self.noNodes, spring=self.spring, \
			noNeighbours=self.noNeighbours, environment=env)
		if self.trainOmega:
			control = SineControl(morph)
		else:
			control = SineControl(morph, omega=self.omega)
		robot = Robot(morph, control)
		plotter = Plotter(plot=False);
		simulenv = SimulationEnvironment(timeStep=self.simTimeStep, simulationLength=int(self.simTime/self.simTimeStep), \
		plot=plotter, perfMetr=self.perfMetr, controlPlot=self.controlPlot, refDist=self.refDist, refPower=self.refPower, \
		noisy = self.noisy)

		# Append training variables
		trainscheme = TrainingScheme()
		if self.trainOmega:
			trainscheme.createTrainVariable("base_omega", 0, self.omega)
		trainscheme.createTrainVariable("phase", 0, 2 * np.pi)
		trainscheme.createTrainVariable("amplitude", 0, self.maxAmplitude)
		trainscheme.createTrainVariable("spring", 0, self.maxSpring)

		# Choose optimization type
		saver = Save(None, self.fileName, self.folderName)
		if self.optMethod == "CMA":
			train = CMATraining(trainscheme, robot, simulenv, saver=saver, maxIter=self.maxIter)
		elif self.optMethod == "Genetic":
			train = GeneticTraining(trainscheme, robot, simulenv, saver=saver, populationSize=self.popSize,\
				noGenerations=self.noGen)
		else:
			train = RandomTraining(trainscheme, robot, simulenv, saver=saver, noInstances=self.maxIter)

		# Perform optimization
		print("\n == Experiment starting with the following parameters == \n" + str(self.__dict__) + "\n")
		param, score, t_tot = train.run() 
		bestRobot = trainscheme.normalizedList2robot(train.bestParameters, robot)

		# Print and save results
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()
		size = comm.Get_size()
		machine = platform.node()
		print("-- " + machine + " (" + str(rank+1) + "/" + str(size) + ")" + \
			" -- Total training time: " + "{:.1f}".format(t_tot)  + " s")
		print("-- " + machine + " (" + str(rank+1) + "/" + str(size) + ")" + \
			" -- Best score (" + self.perfMetr + "): {:.4f}".format(score)  + "\n")
		train.save(savePlot=False)
		print(" == Experiment finished with the following parameters == \n\n" + str(self.__dict__) + "\n")

def createParetoVal(pool_n=1):
	"""Return a 2D list of Amplitude and Omega couple for drawing a pareto curve"""

	omega = [1, 2, 5, 10, 20, 40]
	amplitude =  np.linspace(0.001, 1, 20).tolist()
	liste = []

	for i in range(pool_n):
		for om in omega:
			for am in amplitude:
				liste.append([om, am])

	return liste

def createSimTimeVal():
	"""Return a 2D list of Simulation time and optimization methdd"""

	st = [15]#0.5, 1, 2, 5, 10, 20, 25]
	opt =  ["CMA", "Genetic", "Random"]
	liste = []

	for s in st:
		for o in opt:
			liste.append([s, o])

	return liste

def createKMVal():
	"""Return a 2D list of spring constant and mass value"""

	compliance = np.linspace(0.001, 0.1, num=30).tolist()
	spring = []
	for c in compliance:
		spring.append(float(1)/c)
	#spring = [0.5, 1, 2, 5 20, 50, 100, 200, 500]
	mass =  [1]#[0.1, 0.5, 1, 2, 5, 10]
	liste = []

	for k in spring:
		for m in mass:
			liste.append([k, m])

	return liste

def createNodesVal():
	"""Return a list of nodes number"""

	nodes = []
	# Average with 5 values
	for i in range(1, 5):
		nodes.extend(range(3, 27))
	liste = []

	for n in nodes:
			liste.append([n])

	return liste

def createRefVal(pool_n=1):
	"""Return a 2D list of reference distance and power"""

	dist = [100, 200, 400]
	power =  [2000, 5000]
	liste = []

	for i in range(pool_n):
		for p in power:
			for d in dist:
				liste.append([p, d])

	return liste

def createRefPowerParetoVal():
	"""Return a list of reference power"""

	liste = []
	# power =  np.linspace(100, 20000, num=10).tolist()
	power =  np.array([100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500]).tolist()
	freq =  []
	# Average with 5 values
	for i in range(1, 5):
		freq.extend(range(1, 5))

	for f in freq:
		for p in power:
				liste.append([p, 2*np.pi*f])

	return liste

def createRefDistParetoVal():
	"""Return a list of reference distance"""

	liste = []
	dist =  np.linspace(10, 300, num=10).tolist()
	freq =  []
	# Average with 5 values
	for i in range(1, 5):
		freq.extend(range(1, 5))

	for f in freq:
		for d in dist:
				liste.append([d, 2*np.pi*f])

	return liste

def createNoiseVal():
	"""Return a 2D list of reference distance and power"""

	noise =  np.logspace(-8, 0, num=10)

	return noise

def createOmegaVal():
	""" Return a 2D list of omega values"""
	
	liste = []
	nodes =  [5, 10, 15, 20]
	freq =  []
	# Average with 5 values
	for i in range(1, 3):
		freq.extend(np.logspace(-0.7, 1, num=10).tolist())

	for f in freq:
		for n in nodes:
				liste.append([2*np.pi*f, n])

	return liste
