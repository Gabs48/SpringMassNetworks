#! /usr/bin/env python2

from roboTraining.robot import *
from roboTraining.simulate import *
from roboTraining.training import *
from roboTraining.utils import *

from multiprocessing import *
from mpi4py import MPI
import platform
import sys
import datetime

def experiment(fileName_="CMA", folderName_="Data", noNodes_=20, spring_=100, noNeighbours_=3, plot_=False, \
	simTimeStep_=0.005, simTime_=20, controlPlot_=False, maxIter_=10, maxOmega=10, maxAmplitude=0.25):
	"""Start a standard experiment"""

	env = HardEnvironment()
	morph = SpringMorphology(noNodes=noNodes_ , spring=spring_, noNeighbours=noNeighbours_, environment=env)
	control = SineControl(morph)
	robot = Robot(morph,control)

	plotter = Plotter(plot=False);
	simulenv = SimulationEnvironment(timeStep=simTimeStep_, simulationLength=int(simTime_/simTimeStep_), \
	plot=plotter, controlPlot=controlPlot_)

	trainscheme = TrainingScheme()
	trainscheme.createTrainVariable("omega", 0, maxOmega)
	trainscheme.createTrainVariable("phase", 0, 2 * np.pi)
	trainscheme.createTrainVariable("amplitude", 0, maxAmplitude)

	saver = Save(None, fileName_, folderName_)
	train = CMATraining(trainscheme, robot, simulenv, saver=saver, maxIter=maxIter_)

	param, score, t_tot = train.run() # perform optimization

	process_name = current_process().name
	print("-- " + process_name + " -- Total training time: " + "{:.1f}".format(t_tot)  + " s")
	train.save()


if __name__ == "__main__":
	"""Start the experiment function with different parameters"""

	# Get MPI info
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()
	machine = platform.node()	

	# Print machine info	
	print("\n == Robot Training Experiment with Multiprocessing == ")
	print("\n  Date: " + str(datetime.datetime.now()))
	print("  Machine: " + machine + " (" + str(rank+1) + "/" + str(size) + ")")
	print("  OS: " + str(platform.system()) + " version " + str(platform.release()))
        print("  Python version: " + str(platform.python_version ()+ "\n"))

	# Do experiment
	fileName = "Machine-" + str(rank)
	experiment(fileName_=fileName)
