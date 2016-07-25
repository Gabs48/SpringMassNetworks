#! /usr/bin/env python2

from roboTraining.robot import *
from roboTraining.simulate import *
from roboTraining.training import *
from roboTraining.utils import *

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
		maxAmplitude_=0.25):
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
		self.maxAmplitude = maxAmplitude_

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
		train = CMATraining(trainscheme, robot, simulenv, saver=saver, maxIter=self.maxIter)

		# Perform optimization
		param, score, t_tot = train.run() 

		# Print and save results
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()
		size = comm.Get_size()
		machine = platform.node()
		print("-- " + machine + " (" + str(rank+1) + "/" + str(size) + ")" + \
			" -- Total training time: " + "{:.1f}".format(t_tot)  + " s")
		print("-- " + machine + " (" + str(rank+1) + "/" + str(size) + ")" + \
			" -- Best score (" + self.perfMetr + "): {:.2f}".format(score)  + "\n")
		print(" == Experiment finished with the following parameters == \n\n  " + str(self.__dict__) + "\n")
		train.save()


def createParetoVal():
	"""Return a 2D list of Amplitude and Omega couple for drawing a pareto curve"""

	omega = [1, 2, 5, 10, 20, 40]
	amplitude =  np.linspace(0.001, 1, 20).tolist()
	liste = []

	for om in omega:
		for am in amplitude:
			liste.append([om, am])

	return liste


if __name__ == "__main__":
	"""Start the experiment function with different parameters"""

	trainingIt = 5000
	simTime = 25

	# Get MPI info
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()
	machine = platform.node()

	# Print machine info	
	print("\n == Initializing Robot Training Experiment == ")
	print("\n  Date: " + str(datetime.datetime.now()))
	print("  Machine: " + machine + " (" + str(rank+1) + "/" + str(size) + ")")
	print("  OS: " + str(platform.system()) + " version " + str(platform.release()))
	print("  Python version: " + str(platform.python_version ()))
	print("  Argument List: " +  str(sys.argv) + "\n")

	# Do experiment
	if len(sys.argv) > 1:

		#  Simulate different couple of amplitude and omega to create a pareto curve
		if sys.argv[1].lower() == "pareto":

			# Get arg list and estimate iteration number and time
			arg_list = createParetoVal()
			fileName = "Machine-" + str(rank)
			n_iteration = int(math.ceil(len(arg_list)/float(size)))
			print(" == Running " +  str(len(arg_list)) + " experiments on " + str(size) + \
				" processors: " + str(n_iteration) + " optimizations expected in approximately " + \
				"{:.2f} hours \n".format(float(simTime) / 20 * trainingIt * n_iteration / 3600) + " == ")

			# Simulate multiple time if the number of cores does not correspond to number of points
			for i in range(n_iteration):
				index = i * size + rank
				if index < len(arg_list):
					print("-- " + machine + " (" + str(rank+1) + "/" + str(size) + ")" + \
						" -- Experiment " + str(index+1) + " with Omega=" + \
						str(arg_list[index][0]) + " and Amplitude=" + str(arg_list[index][1]))
					e = Experiment(fileName_=fileName, folderName_="Pareto", maxOmega_=arg_list[index][0], \
						simTime_=simTime, maxIter_=trainingIt,  maxAmplitude_=arg_list[index][1])
					e.run()
	else:
		# Simulate a pool of CMA otpimization with the same arguments
		print(" == Running " +  str(size) + " experiments on " + str(size) + \
			" processors: 1 optimization expected in approximately " + \
			"{:.2f} hours == \n".format(float(simTime) / 20 * trainingIt / 3600))
		fileName = "Machine-" + str(rank)

		e = Experiment(fileName_=fileName, folderName_="CMA", simTime_=simTime, maxIter_=trainingIt)
		e.run()
