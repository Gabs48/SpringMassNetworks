from roboTraining.unitTest import *
from roboTraining.training import *
from roboTraining.simulate import *
from roboTraining.robot import *
from roboTraining.utils import *

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import pyplot as plt
import matplotlib
import time
import numpy as np

ENV_ControlSet = False;
ENV_Control = None

def reservoir(name, Lin = False, control = '', neural = False, target = TargetFunctions.sumLength, plot = True,
	replication = False, delay = False, quick = False, optimMorph = False, partial = False, resultPlot = False, iter = 0, saver = None):
	# IMPORTANT PARAMETERS
	spring = 10
	mass = 0.1
	nonLinRatio = 10
	replicationNumber = 5
	delayTime = 4  # 2ms in case of 0.005 timestep
	delayStep = 3
	relNoise = 0.01
	fraction = 0.30

	if not replication: replicationNumber = 1
	if not delay: delayStep = 1
	if not partial: fraction = 1
	if neural: mapping = NeuralNetwork(maxNoEpochs = 500)
	else: mapping = LinearMap()

	if quick: simulationLength = 5000
	else: simulationLength =  40000

	# Robot + Simulation
	env = HardEnvironment()
	if Lin: morph = SpringMorphology(noNodes = 20, spring = spring, noNeighbours = 3, mass = mass, environment = env)
	else: morph = NonLinearMorphology(noNodes = 20, spring = spring, noNeighbours = 3, mass = mass, environment = env, nonLinRatio = nonLinRatio)
	if optimMorph:
		morph.spring = np.loadtxt(open("optimizedMorphology.csv","rb"),delimiter=";")

	# Control
	if ENV_ControlSet:
		control = ENV_Control
	else:
		control = ReservoirControl(morph)
		control.setReplRandSin(amplitude = 0.2, minMod = 2.0, maxMod = 3.0, replicationNumber = replicationNumber, fraction = fraction)
		global ENV_Control
		ENV_Control = control
		global ENV_ControlSet
		ENV_ControlSet = True

	robot = Robot(morph, control)
	if saver is None:
		plotter = Plotter(movie = False, plot = plot, plotCycle = 500, color = False)
	else:
		plotter = Plotter(movie = True, plot = True, plotCycle = 500, movieName = saver.generateName("reservoir" + str(iter)) )
	simulEnv = SimulationEnvironment(timeStep = 1.0 / 200, simulationLength = simulationLength, plot = plotter, controlPlot = plot)

	resSimul = ReservoirSimulation(simulEnv,robot, np.arange(robot.getNoConnections()), 1, target, relNoise = relNoise, mapping = mapping,
			delayStep = delayStep, delayTime = delayTime) 

	# run simulation
	resSimul.runSimulation();
	plt.close()
	# create plots
	if plot or resultPlot:
		fig, ax = resSimul.plot()
		if plot:
			Plot.save2eps(fig, name)

		print resSimul.trainRelError
		print resSimul.testRelError
	return resSimul.trainRelError, resSimul.testRelError

def finalTest(quick = False):
	kwargs = []
	
	# SUM
	#kwargs.append({'neural': False, 'target' : TargetFunctions.sumLength, 'optimMorph' : False, 'partial' : True}) # GOOD
	#kwargs.append({'neural': False, 'target' : TargetFunctions.sumLength, 'optimMorph' : True, 'partial' : True}) # LESS GOOD
	#kwargs.append({'neural': False, 'target' : TargetFunctions.sumLength, 'optimMorph' : False, 'partial' : True, 'Lin': True}) # GOOD

	## QUADRATRIC
	#kwargs.append({'neural': False, 'target' : TargetFunctions.squaredLength, 'optimMorph' : False, 'partial' : True}) #BAD
	#kwargs.append({'neural': False, 'target' : TargetFunctions.squaredLength, 'optimMorph' : True, 'partial' : True}) #BAD

		#Neural
	kwargs.append({'neural': True, 'target' : TargetFunctions.squaredLength, 'optimMorph' : False, 'partial' : True}) #BAD
	#kwargs.append({'neural': True, 'target' : TargetFunctions.squaredLength, 'optimMorph' : True, 'partial' : True}) #BAD
	
	## MEMORY (half second)
	#kwargs.append({'neural': False, 'target' : TargetFunctions.memoryHalfSecond, 'optimMorph' : True, 'partial' : True}) # Relatively Good
	#kwargs.append({'neural': False, 'target' : TargetFunctions.memoryHalfSecond, 'optimMorph' : True, 'partial' : True}) # Unknown
	
	## Replication
	#kwargs.append({'neural': False, 'target' : TargetFunctions.sumLength, 'optimMorph' : False, 'partial' : True, 'replication' : True}) # GOOD
	#kwargs.append({'neural': False, 'target' : TargetFunctions.sumLength, 'optimMorph' : False, 'partial' : True, 'delay' : True}) # GOOD

	# ----------- CORE -----------
	saver = Save(kwargs, 'reservoir', directory = 'Experiments/Reservoir')
	scores = np.zeros((len(kwargs),2))
	for i in range(len(kwargs)):
		trainScore, testScore = reservoir('default', quick = quick , plot = False, resultPlot = True, iter = i, saver = saver, **kwargs[i])
		scores[i, 0] = trainScore
		scores[i, 1] = testScore
		kwargs[i]['train'] = trainScore
		kwargs[i]['test'] = testScore
		plt.savefig(saver.generateName('plot' + str(i),'.png'), format = 'png', dpi = 300)
		plt.close()
	saver.save(scores)


	print scores


def neuralTest():
	x = np.array([np.arange(100)])
	y = np.sqrt(x ** 2 + 5)
	neural = NeuralNetwork(maxNoEpochs = 1000, max = 100)
	self, trainOutput, trainError, trainRelError = neural.train(x,y)
	print trainRelError
	print trainOutput
	plt.plot(x[0],y[0],label = 'target')
	plt.plot(x[0],trainOutput[0], label = 'train')
	plt.show(block = True)


def stepTest():
	env = HardEnvironment()
	morph = NonLinearMorphology(noNodes = 20, spring = 10, noNeighbours = 3, mass = 0.1, environment = env, nonLinRatio = 10)
	control = StepControl(morph, 0.1, 5, fraction = 0.3)
	robot = Robot(morph, control)
	plotter = Plotter(movie = False, plot = True, plotCycle = 10, color = False)
	simulEnv = SimulationEnvironment(timeStep = 1.0 / 200, simulationLength = 2000, plot = plotter, controlPlot = True)
	simul = VerletSimulation(simulEnv,robot,reset = True)
	lengths = []
	for i in range(2000):
		simul.simulateStep()
		simulEnv.plot.update(robot,i)
		lengths.append(robot.getSumOfSpringLengths())
	fig, ax = Plot.initPlot()
	ax.plot(np.arange(len(lengths))/ 200.0 - 5, lengths, 'k.')
	ax.set_xlim([-2,5])
	ax.xaxis.set_major_locator(MultipleLocator(1))
	ax.xaxis.set_minor_locator(MultipleLocator(0.25))
	ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
	ax.xaxis.grid(True, which = 'both') # vertical lines

	Plot.configurePlot(fig,ax,'time', 'sum of Spring Lenghts [m]')
	plt.show()
	Plot.save2eps(fig,"stepResponse")
"""
for i in range(4):
	finalTest(quick = False)
	ENV_ControlSet = False
"""

#stepTest()
finalTest(quick = False)
#neuralTest()