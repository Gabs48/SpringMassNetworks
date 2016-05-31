from roboTraining.unitTest import *
from roboTraining.training import *
from roboTraining.simulate import *
from roboTraining.utils import *
from roboTraining.robot import *

from matplotlib import pyplot as plt
import matplotlib
import time

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

"""
CONTENT

sinTest()
	IMPORTANT: shows how right timestep should be chosen and the superiority of verlet

trainRobot()
	IMPORTANT: trains a robot to good parameter specifications and makes of the trained robot a movie

"""

def morphPlot3D():
	env=HardEnvironment(threeD = True)
	morph=SpringMorphology3D(noNodes = 20 ,spring = 1000, noNeighbours = 4, environment = env)
	fig, ax = morph.plot()
	Plot.save2eps(fig, "3Dmorph")

def morphPlot2D():
	env= HardEnvironment()
	morph=SpringMorphology(noNodes = 20 ,spring = 1000, noNeighbours = 4, environment = env)
	fig, ax = morph.plot()
	Plot.save2eps(fig, "2Dmorph")

def morphPlot2DSpring():
	env= HardEnvironment()
	morph=SpringMorphology(noNodes = 20 ,spring = 1000, noNeighbours = 2, environment = env)
	fig, ax = morph.plot(spring = True)
	Plot.save2eps(fig, "2DmorphSpring")
	

def sinTest():
	robot = emptyRobot(spring = 0.5, ground = True)
	timeStepList = [1.0/1000, 1.0 / 100, 1.0/10]
	sample = [100, 10, 1]
	fig = plt.figure(figsize=(10, 4),dpi = 150)
	ax = plt.gca()
	color = ['r','b','k']
	index = 0
	for timeStep in timeStepList:
		for verlet in [True, False]:
			IterationSteps = int (20 * np.pi / timeStep)
			setState2D(robot, xpos = [0, 1.5], ypos = [1, 1])
			simulation = simpleSimulation(robot, timeStep, IterationSteps, verlet = verlet)
			times = np.zeros(IterationSteps)
			pos = np.zeros(IterationSteps)

			for t in range(IterationSteps):
				simulation.simulateStep()
				simulation.simulEnv.plot.update(simulation.robot,simulation.iterationNumber)
				times[t] = t * timeStep
				pos[t] = robot.state.pos.x[1] - robot.state.pos.x[0]

			sampleTimes = times[::sample[index]]
			samplePos = pos[::sample[index]]
			if verlet:
				plt.plot(sampleTimes, samplePos, '.', alpha = 0.8, color = color[index], markersize = 3, label = r'$\Delta t = $ %(num)s $\omega^{-1} $ Verlet' % {'num': num2str(timeStep)} )
			else:
				plt.plot(sampleTimes, samplePos, color = color[index], label = r'$\Delta t = $ %(num)s $\omega^{-1} $  Standard' % {'num': num2str(timeStep)} )
			print IterationSteps
		index += 1
	plt.ylim(0,2)
	Plot.configurePlot(fig, ax, "time", "spring length", size = 'small')
	plt.show()

	fig2 = plt.figure(figsize=(10, 4),dpi = 150)
	ax2 = plt.gca()

	for timeStep in [1.0/30, 1.0/10, 1.0/3]:
		IterationSteps = int (20 * np.pi / timeStep)
		setState2D(robot, xpos = [0, 1.5], ypos = [1, 1])
		simulation = simpleSimulation(robot, timeStep, IterationSteps, verlet = True)
		times = np.zeros(IterationSteps)
		pos = np.zeros(IterationSteps)

		for t in range(IterationSteps):
			simulation.simulateStep()
			simulation.simulEnv.plot.update(simulation.robot,simulation.iterationNumber)
			times[t] = t * timeStep
			pos[t] = robot.state.pos.x[1] - robot.state.pos.x[0]
		plt.plot(times, pos, label = r'$\Delta t = $ %(num)s $\omega^{-1} $  Verlet' % {'num': num2str(timeStep)} )
	plt.ylim(0.4,1.6)
	Plot.configurePlot(fig2, ax2, "time", "spring length", size = 'small')
	Plot.save2eps(fig, "InitTestSineWaves")
	Plot.save2eps(fig2, "InitTestSineWaves2")

	plt.show(block = True)

def trainRobot():
	env = HardEnvironment()
	morph = SpringMorphology(noNodes = 20,spring = 100, environment = env)
	control = SineControl(morph,amplitude = 0.5) 
	robot = Robot(morph, control)
	#plotter = Plotter(movie = True, plot = True, movieName = "RobotHeavyOscillations")
	plotter = Plotter(movie = False, plot = False, plotCycle = 500)
	refPower = robot.referencePower(100, 1, 0.2, 5)
	simulEnv = SimulationEnvironment(timeStep = 1.0/200, simulationLength = 120, plot = plotter, controlPlot = False, perfMetr = 'powereff', refPower = refPower, refDist = 10) # 00

	trainscheme = TrainingScheme()
	trainscheme.createTrainVariable("phase", 0 , 2 * np.pi)
	trainscheme.createTrainVariable("amplitude", 0 , 0.25)
	trainscheme.createTrainVariable("omega", 0 , 10)
	#SpringMorphology.checkConfig(maxSpring, 1, timeStep)
	train = GeneticTraining(trainscheme, robot, simulEnv , noGenerations = 50, populationSize = 30)
	train.run();
	train.save();
	bestRobot = trainscheme.normalizedMatrix2robot(train.bestParameters, robot)
	print(train.bestParameters)
	print(train.optimalscore)

	plotter = Plotter(movie = True, plot = True, movieName = "Robot13April", plotCycle = 3)
	simulEnv = SimulationEnvironment(timeStep = 1.0/200, simulationLength = 120, plot =  plotter) # 00
	simul = VerletSimulation(simulEnv, robot)
	print simul.runSimulation();

#trainRobot()
#morphPlot3D()
#sinTest()

morphPlot2DSpring()