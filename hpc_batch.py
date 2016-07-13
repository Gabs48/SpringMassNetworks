#!/usr/bin/python2

from multiprocessing import *

from roboTraining.robot import *
from roboTraining.simulate import *
from roboTraining.training import *
from roboTraining.utils import *


def experiment(fileName="CMA", folderName="Data", noNodes_=20, spring_=100, noNeighbours_=3, plot_=False, \
	simTimeStep_=0.005, simTime_=0.5, controlPlot_=False, maxIter_=10, maxOmega=10, maxAmplitude=0.25):
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

	saver = Save(None, fileName, folderName)
	train = CMATraining(trainscheme, robot, simulenv, saver=saver, maxIter=maxIter_)

	param, score, t_tot = train.run() # perform optimization

	process_name = current_process().name
	print("-- " + process_name + " -- Total training time: " + "{:.1f}".format(t_tot)  + " s")
	train.save()


if __name__ == "__main__":
	"""Batch multiprocessing loop"""

	p_list = []

	for i in range(20):
		fileName = "Process-" + str(i+1)
		p = Process(target=experiment, args=(fileName, ), name="P-" + str(i+1))
		p.start()
		p_list.append(p)
		
	for p in p_list:
		p.join()



