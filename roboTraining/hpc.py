
from robot import *
from simulate import *
from training import *
from utils import *

import jsonpickle
import math
from multiprocessing import *
from mpi4py import MPI
import platform
import sys
import datetime


class Experiment(object):
	"""Class to perform standard experiments"""

	def __init__(self, fileName_="CMA", folderName_="Data", noNodes_=20, spring_=100, noNeighbours_=3, plot_=False, \
		simTimeStep_=0.005, simTime_=20, perfMetr_="dist", controlPlot_=False, maxIter_=5000, maxOmega_=10, \
		optMethod_="CMA", maxAmplitude_=0.25, popSize_=30):
		"""Initialize the variables lists"""

		self.fileName = fileName_
		self.folderName = folderName_
		self.noNodes = noNodes_
		self.spring = spring_
		self.noNeighbours = noNeighbours_
		self.plot = plot_
		self.simTimeStep = simTimeStep_
		self.simTime = simTime_
		self.perfMetr = perfMetr_
		self.controlPlot = controlPlot_
		self.maxIter = maxIter_
		self.maxOmega = maxOmega_
		self.optMethod = optMethod_
		self.maxAmplitude = maxAmplitude_
		self.popSize = popSize_
		self.noGen = int(self.maxIter / self.popSize)

	def run(self):
		"""Run the experiment"""

		# Init environment
		env = HardEnvironment()
		morph = SpringMorphology(noNodes=self.noNodes , spring=self.spring, noNeighbours=self.noNeighbours, environment=env)
		control = SineControl(morph)
		robot = Robot(morph, control)

		plotter = Plotter(plot=False);
		simulenv = SimulationEnvironment(timeStep=self.simTimeStep, simulationLength=int(self.simTime/self.simTimeStep), \
		plot=plotter, perfMetr=self.perfMetr, controlPlot=self.controlPlot)

		trainscheme = TrainingScheme()
		trainscheme.createTrainVariable("omega", 0, self.maxOmega)
		trainscheme.createTrainVariable("phase", 0, 2 * np.pi)
		trainscheme.createTrainVariable("amplitude", 0, self.maxAmplitude)

		saver = Save(None, self.fileName, self.folderName)
		if self.optMethod == "CMA":
			train = CMATraining(trainscheme, robot, simulenv, saver=saver, maxIter=self.maxIter)
		elif self.optMethod == "Genetic":
			train = GeneticTraining(trainscheme, robot, simulenv, saver=saver, populationSize=self.popSize,\
				noGenerations=self.noGen)
		else:
			train = RandomTraining(trainscheme, robot, simulenv, saver=saver, noInstances=self.maxIter)

		# Perform optimization
		param, score, t_tot = train.run() 
		bestRobot = trainscheme.normalizedMatrix2robot(train.bestParameters, robot)
		print(train.bestParameters)
		print(train.optimalscore)

		pltt = Plotter(movie = True, plot = True, movieName = "Robot13April", plotCycle = 6, color=True)
		simulEnv = SimulationEnvironment(timeStep = 1.0/200, simulationLength = 400, plot =  pltt) # 00
		simul = VerletSimulation(simulEnv, robot)
		print simul.runSimulation();

		jsonpickle.set_encoder_options('simplejson', sort_keys=True, indent=4)
		with open("simulEnv_orig.json", 'wb') as f:
			f.write(jsonpickle.encode(simulenv))
			f.close()
		with open("robot_orig.json", 'wb') as f:
			f.write(jsonpickle.encode(robot))
			f.close()

		# Print and save results
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()
		size = comm.Get_size()
		machine = platform.node()
		print("-- " + machine + " (" + str(rank+1) + "/" + str(size) + ")" + \
			" -- Total training time: " + "{:.1f}".format(t_tot)  + " s")
		print("-- " + machine + " (" + str(rank+1) + "/" + str(size) + ")" + \
			" -- Best score (" + self.perfMetr + "): {:.2f}".format(score)  + "\n")
		train.save()
		print(" == Experiment finished with the following parameters == \n\n  " + str(self.__dict__) + "\n")

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

	st = [0.5, 1, 2, 5, 10, 20, 25, 50]
	opt =  ["CMA", "Genetic", "Random"]
	liste = []

	for s in st:
		for o in opt:
			liste.append([s, o])

	return liste

