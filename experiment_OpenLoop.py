import matplotlib.pyplot as plt
import numpy as np
import os
from multiprocessing import Pool, freeze_support

from scipy.optimize import fsolve
from roboTraining.training import *
from roboTraining.simulate import *
from roboTraining.robot import *
from roboTraining.utils import *
from functools import partial

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

plotSize = 'normal'
# _________________________________________________ BASIC TESTS __________________________________________________________________
def CMAConvergence(load = False, long = False):
	if long:
		name = 'CMAconvergenceLong'
	else:
		name = 'CMAconvergence'
	folder = "Experiments"
	if load is False:
		training = trainRobot(CMA = True, quickTest= False)
		sigma = training.sigmaList
		iter = np.arange(*np.shape(sigma)) * training.popSize
		saver = Save(np.vstack((iter, sigma)), name, directory = folder, floatFormat = '.5e')
		saver.save()
	elif long:
		iter, sigma= loadData("Experiments/Final/CMAconvergenceLongconfig.csv")
	else:
		iter, sigma= loadData("Experiments/Final/CMAconvergenceconfig.csv")

	if long:
		data = np.genfromtxt('Experiments/Final/cmavsgaLong.csv', delimiter=';')
		markersize = 2
	else:
		data = np.genfromtxt('Experiments/Final/cmavsga.csv', delimiter=';')
		markersize = 4
	resultsCMA = data[0,:]
	resultsGA = data[1,:]
	x = np.arange(len(resultsCMA))

	fig, ax = Plot.initPlot()
	ax.plot(x, resultsCMA, 'b.', label = 'CMA-ES', markersize = markersize)
	ax.plot(x, resultsGA, 'r.', label = 'GA', markersize = markersize)
	
	ax2 = ax.twinx()
	ax2.plot(iter, sigma, 'k-')
	ax2.set_yscale('log')
	Plot.configurePlot(fig, ax2, "iteration", r"step size $\sigma$", legend = False)
	Plot.configurePlot(fig, ax, 'iteration', 'Distance Traveled [m]', legend = True, legendLocation = 'lower center')
	if load is True:
		Plot.save2eps(fig, name)
	else: plt.show(block = True)

def specificRunPlot(long = False):
	"""config uit GA Training run0 en CMATraining run0:	
	20 nodes 3 neighbours voor 10 sec laten evalueren"""
	if long:
		data = np.genfromtxt('Experiments/Final/cmavsgaLong.csv', delimiter=';')
		markersize = 2
	else:
		data = np.genfromtxt('Experiments/Final/cmavsga.csv', delimiter=';')
		markersize = 4
	resultsCMA = data[0,:]
	resultsGA = data[1,:]
	x = np.arange(len(resultsCMA))
	fig, ax = Plot.initPlot()
	ax.plot(x, resultsCMA, 'b.', label = 'CMA-ES', markersize = markersize)
	ax.plot(x, resultsGA, 'r.', label = 'GA', markersize = markersize)
	Plot.configurePlot(fig, ax, 'iteration', 'Distance Traveled [m]', legend = True)
	plt.show(block = True)
	Plot.save2eps(fig, 'GAvsCMA')

def longTermExperiment(loadFile = None):
	name = "longTermTraining"
	folder = "Experiments"
	if loadFile is None:
		score = trainRobot(quickTest= False, name = name, longTest = True)
	#required plot is inside the GA folder

def movieMaker():
	trainRobot(movie = True, name = 'movieMaker')

def frameMaker():
	quickTest = False
	score, bestRobot = trainRobot(fullOutput = True, quickTest = quickTest)
	plotter = Plotter(movie = False, plot = False)
	simulationLength = 200
	simulEnv = SimulationEnvironment(timeStep = 1.0/200, simulationLength = simulationLength, plot = plotter, controlPlot = False)
	bestRobot.reset()
	confirm = False
	for i in range(11):
		fig, ax = bestRobot.staticPlot()
		Plot.configurePlot(fig, ax, "x [m]", "y [m]", legend = False, huge = True)
		confirm = Plot.save2eps(fig, "animationSec" + str(i), confirm = confirm )
		simul = VerletSimulation(simulEnv, bestRobot, reset = False)
		simul.runSimulation()

# _________________________________________________ RIGID vs FLEXIBLE __________________________________________________________________
def springExperiment(loadFile = None):
	name = "SpringExperiment"
	springArray = [10, 20, 50, 100, 200, 400]
	experiment(name, loadFile , "Spring Strength [N/m]", "Distance Traveled [m] ", "spring",  springArray)
	
def springExperimentConstantPower(loadFile = None):
	name = "SpringExperimentConstantPower"
	springArray = np.array([10, 20, 50, 100, 200, 400])
	maxAmplitude = 0.2 * np.sqrt(100.0 / springArray)
	experiment(name, loadFile , "Spring Strength [N/m]", "Distance Traveled [m]", "spring",  springArray, "maxAmplitude", maxAmplitude )

def springExperimentMass(loadFile = None):
	name = "SpringExperimentMass"
	springArray = np.array([10, 20, 50, 100, 200, 400])
	maxAmplitude = 0.2 * np.sqrt(100.0 / springArray)
	mass = springArray / 100.0
	experiment(name, loadFile , "Spring Strength [N/m]", "Distance Traveled [m]", "spring",  springArray, "maxAmplitude", maxAmplitude, "mass", mass )

def springExperimentMassNonLinear(loadFile = None, nonLin = 1):
	name = "SpringExperimentMassNonLinear" + str(nonLin)
	springArray = np.array([10, 20, 50, 100])
	maxAmplitude = np.zeros(4)
	

	for i in range(4):
		spring = springArray[i]
		def powerDiff(amplitude):
			return amplitude ** 2 * (1 + nonLin * amplitude ** 2) * spring - 0.2 ** 2 * 100
		maxAmplitude[i] = fsolve(powerDiff, 0.2 * np.sqrt(100.0 / spring))

	mass = springArray / 100.0
	experiment(name, loadFile , "Spring Strength [N/m]", "Distance Traveled [m]", "spring",  springArray, "maxAmplitude", maxAmplitude, "mass", mass ,"nonLin",nonLin)

def plotSpringExperimentMassNonLinear():
	name0 = "Experiments/Final/SpringExperimentMassNonLinear0config.csv"
	name2 = "Experiments/Final/SpringExperimentMassNonLinear2config.csv"
	name10 = "Experiments/Final/SpringExperimentMassNonLinear10config.csv"
	indices, results0 = loadData(name0)
	indices, results2 = loadData(name2)
	indices, results10 = loadData(name10)
	fig, ax  = Plot.initBarPlot(indices, results0, "","", label = r'$\alpha = 0$', barWidth = 0.3)
	Plot.addbars(ax, results2,'#555555', label = r'$\alpha = 2$', barWidth = 0.3, rank = 1)
	Plot.addbars(ax, results10,'#ff5555', label = r'$\alpha = 10$', barWidth = 0.3, rank = 2)
	Plot.configurePlot(fig, ax, "Spring Strength [N/m]", "Distance Traveled [m]")
	Plot.save2eps(fig,'nonLinearExperiment')

def powerEffspring(loadFile = None):
	name = "powereffSpring"
	springArray = np.array([10, 20, 50, 100, 200, 400])
	maxAmplitude = 0.2 * np.sqrt(100.0 / springArray)
	mass = springArray / 100.0
	distRef = 100
	fracNomPower = 0.3
	powerRef = (60 * (2 * 0.2) ** 2 * 100 * 10) * fracNomPower
	experiment(name, loadFile , "Spring Strength [N/m] ", "Energy Efficiency Score", "spring",  springArray, "maxAmplitude", maxAmplitude, "mass", mass,
				"perf", 'powereff', "powerRef", powerRef, "distRef", distRef )

def partialExperiment(loadFile = None):
	name = "partialExperiment"
	trainFraction = np.array([0.15, 0.20]) #np.array([0.1, 0.25, 0.5, 0.75, 0.9, 1])
	maxAmplitude = 0.2 / np.sqrt(trainFraction)
	experiment(name, loadFile, " Train Fraction ", "Distance Traveled [m]", 'trainFraction', trainFraction, 'maxAmplitude', maxAmplitude)

# _________________________________________________ OPTIMIZED MORPHOLOGY __________________________________________________________________
def springStrengthsExperiment(loadFile = None):
	name = "optimizeSpringStrengths"
	folder = 'Experiments'
	_, bestRobot = trainRobot(quickTest = False, name = name, optimizeSprings = True, fullOutput = True, longTest = True)
	bestRobot.morph.strengthPlot()

def springLengthsExperiment(loadFile = None):
	name = "optimizeSpringLengths"
	folder = 'Experiments'
	_, bestRobot = trainRobot(quickTest = False, name = name, optimizeSprings = True, fullOutput = True, longTest = True)
	bestRobot.morph.strengthPlot() # bad: better use first relaxation!!!)

# _________________________________________________ COMPLEXITY __________________________________________________________________
def noNodesExperiment(loadFile = None):
	name = "NoNodes"
	noNodes = np.array([10, 15, 20, 30, 50])
	experiment(name, loadFile, " Number of Nodes ", "Distance Traveled [m]", 'noNodes', noNodes)

# _________________________________________________ OTHER __________________________________________________________________
def paretoCurve(loadFile = None, maxOmega = 10):
	name = "paretoCurve" + str(maxOmega)
	if maxOmega >= 7:
		amplitude = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
	else: amplitude = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
	mass = 0.2
	if loadFile is None:
		experiment(name, loadFile, "Modulation Amplitude", "Distance Traveled", 'maxAmplitude',amplitude, 'spring', 20, 'mass', mass, 'maxOmega', maxOmega)
	else:
		indices, results = loadData(loadFile)
		fig, ax  = Plot.initBarPlot(indices, results, "Modulation Amplitude", "Distance Traveled")
		mean = np.average(results, axis = 1) 
		powereff = mean /amplitude ** 2 / mean[-1] * amplitude[-1] ** 2
		index = np.arange(*np.shape(mean)) + 0.25

		ax2 = ax.twinx()
		ax2.plot(index, powereff, 'kx', markersize = 15, mew = 5, label = 'power efficiency')
		Plot.configurePlot(fig, ax2, "Modulation Amplitude", "Relative Power Efficiency")
		ax2.set_ylim([0,3])
		plt.show()
		Plot.save2eps(fig, name)

def plotPareto():
	name =  ["Experiments/Final/paretoCurve1config.csv", "Experiments/Final/paretoCurve2config.csv", "Experiments/Final/paretoCurve5config.csv", "Experiments/Final/paretoCurveconfig.csv", "Experiments/Final/paretoCurve20config.csv"]
	omegaRes = np.sqrt(20 / 0.2) # Resonance frequency of one mass spring system
	maxOmegaList =[1, 2, 5, 10, 20]
	noTest = len(name)
	power0 = 0;
	fig, ax = Plot.initPlot()
	minP = 0.01 ** 2 
	maxP = 3
	minY = 0.1 ** 5 
	maxY = 3

	indices, results = loadData(name[noTest - 1])
	dist0 = np.average(results, axis = 1)[-1]
	power0 = maxOmegaList[noTest - 1] * indices[-1] ** 2

	for i in range(noTest):
		indices, results = loadData(name[i])
		omega = maxOmegaList[i]
		meanDist = np.average(results, axis = 1)
		normPower =  indices ** 2 * omega / power0
		normDist = meanDist / dist0
		ax.loglog(normPower, normDist, '.', label = r'$\omega_{max}\ =  \ $' + num2str(omega / omegaRes) + r'$ \omega_{res}$', markersize = 20)

	for powerEff in np.logspace(-5,5,30):
		x = np.array([1e-5, 10])
		plt.plot(x, powerEff *x, 'k--', alpha = 0.5)
	Plot.configurePlot(fig, ax, 'relative power','relative speed', legendLocation = 'lower center')
	ax.set_xlim([minP, maxP])
	ax.set_ylim([minY, maxY])
	plt.show(block = False)
	Plot.save2eps(fig,'paretoEnergy')

def noisySimulation(loadFile = None, noNodes = 20, spring = 100, maxAmplitude = 0.25, maxOmega = 10, mass = 1, quickTest = False):
	name = 'noisySimulation'
	folder = "Experiments"
	if loadFile is None:
		_ , bestRobot = trainRobot(fullOutput = True, quickTest= False)
		simulationLength = 2000
		noPoints = 1000
		noiseArray = np.logspace(-8,-1,num = noPoints)
		distanceArray = np.zeros(noPoints)
		for i in range(noPoints):
			noise = noiseArray[i]
			simulEnv = SimulationEnvironment(timeStep = 1.0/200, simulationLength = simulationLength, controlPlot = False)
			simul = NoisyVerletSimulation(simulEnv, bestRobot, noise =  noise, reset = True)
			distanceArray[i] = simul.runSimulation();

		saver = Save(np.vstack((noiseArray, distanceArray)), name, directory = folder, floatFormat = '.10f')
		saver.save()
	else:
		noiseArray, distanceArray= loadData(loadFile)

	avergeArray = movingAverage(distanceArray, 101)
	fig, ax = Plot.initPlot()
	ax.semilogx(noiseArray, distanceArray, 'k.', label = "results" )
	ax.semilogx(noiseArray, avergeArray, 'b-', label = "moving average")
	Plot.configurePlot(fig, ax, "relative noise" + r'$ \ \sigma$', "Distance Traveled [m]", legend = False)
	ax.set_ylim(ymin = 0)
	if loadFile is not None:
		Plot.save2eps(fig, name)
	else: plt.show(block = True)

# _________________________________________________ BASIS __________________________________________________________________

def experiment(name, loadFile = None, xlabel = "", ylabel = "", *args):
	folder = "Experiments"
	if __name__ == '__main__':
		if loadFile is not None:
			indices, results = loadData(loadFile)
		else:
			freeze_support()
			pool = Pool(3)
			noReplications = 6
			if len(args) > 0:
				assert len(args) % 2 == 0
				noParams = len(args)/2
				noTests = len(args[1])
				names = [""] * noParams
				params = [[]] * noParams
				for i in range( noParams):
					names[i] = args[2 * i]
					params[i] = args[2 * i + 1]
				results = np.zeros((noTests, noReplications)) 

				for i in range(noTests):
					kwargs = {}
					for k in range(noParams):
						kwargs[ names[k] ] = getItemAt(params[k], i)
					mapfunc = partial(trainRobot, quickTest = False, movie = False, **kwargs)
					results[i, :] =  pool.map(mapfunc, np.arange(noReplications))
			saver = Save(np.vstack(([params[0]], results.T)), name, directory = folder)
			saver.save()
			indices = params[0]
		fig, ax  = Plot.initBarPlot(indices, results, xlabel, ylabel)
		Plot.configurePlot(fig, ax, xlabel, ylabel, legend= False, size = plotSize)
		if loadFile is not None:
			plt.show()
			Plot.save2eps(fig, name)
		else: plt.show(block = False)

def loadData(loadFile):
	"""loadFile is relativePath (do not forget to use only / instead of \ ) """
	data = np.genfromtxt(loadFile, delimiter=';')
	labels = data[0,:]
	results = data[1:, :].T
	return labels, results

def trainRobot(iterNum = 0,noNodes = 20, spring = 100, maxAmplitude = 0.25, maxOmega = 10, perf = 'dist', powerRef = None, distRef = None,
					mass = 1, quickTest = False, movie = False, trainFraction = None, fullOutput= False, longTest = False, name = 'default',
				optimizeSprings = False, optimizeRestLengths = False, CMA = False, nonLin = 0):
	env = HardEnvironment()
	if nonLin == 0:
		morph = SpringMorphology(noNodes = noNodes,spring = spring, environment = env, mass = mass)
	else:
		morph = NonLinearMorphology(noNodes = noNodes,spring = spring, environment = env, mass = mass, nonLinRatio = nonLin)
	control = SineControl(morph,amplitude = 0.5) 
	robot = Robot(morph, control)
	plotter = Plotter(movie = False, plot = False)
	maxLength = 3
	if quickTest:
		simulationLength = 40
		noGenerations = 5
		populationSize = 5
	elif longTest:
		simulationLength = 2000
		noGenerations = 200
		populationSize = 50
	else:
		simulationLength = 2000
		noGenerations  = 50
		populationSize = 30
	if perf == 'dist':
		simulEnv = SimulationEnvironment(timeStep = 1.0/200, simulationLength = simulationLength, plot = plotter, controlPlot = False)
	elif perf == 'powereff':
		assert powerRef is not None and distRef is not None, 'no reference power or distance is given'
		simulEnv = SimulationEnvironment(timeStep = 1.0/200, simulationLength = simulationLength, plot = plotter, controlPlot = False, 
			perfMetr = 'powereff', refDist = distRef, refPower = powerRef)
	else:
		 raise NotImplementedError, 'the requested performance metric is not implemented: ' + perf
	if trainFraction is None:
		trainscheme = TrainingScheme()
		trainscheme.createTrainVariable("phase", 0 , 2 * np.pi)
		trainscheme.createTrainVariable("amplitude", 0 , maxAmplitude)
		trainscheme.createTrainVariable("omega", 0 , maxOmega)
		if optimizeSprings:
			trainscheme.createTrainVariable("spring", 0, spring)
		if optimizeRestLengths:
			trainscheme.createTrainVariable("spring", 0, spring)
			trainscheme.createTrainVariable("restLength", 0, maxLength)
	else:
		assert optimizeSprings is False and optimizeRestLengths is False
		trainscheme = PartialTrainingScheme(robot = robot, fraction = trainFraction)
		trainscheme.createTrainVariable("phase", 0 , 2 * np.pi, zero = True)
		trainscheme.createTrainVariable("amplitude", 0 , maxAmplitude, zero = True)
		trainscheme.createTrainVariable("omega", 0 , maxOmega, zero = True)
	if not CMA:
		saver = Save(None, name, 'GA') 
		train = GeneticTraining(trainscheme, robot, simulEnv , noGenerations = noGenerations, populationSize = populationSize, saver = saver, databaseName = "GA" + str(iterNum))
	else:
		saver = Save(None, name, 'CMA') 
		train = CMATraining(trainscheme, robot, simulEnv , maxIter = 10000, saver = saver)
		
	
	train.run();
	if not quickTest:
		train.save();
	bestRobot = trainscheme.normalizedMatrix2robot(train.bestParameters, robot)
	if movie:
		plotter = Plotter(movie = True, plot = True, movieName = "TrainedRobot2", plotCycle = 6)
		simulEnv = SimulationEnvironment(timeStep = 1.0/200, simulationLength = simulationLength, plot =  plotter)
		simul = VerletSimulation(simulEnv, robot)
		print simul.runSimulation();
	if fullOutput:
		return train.optimalscore, bestRobot
	elif CMA:
		return train
	else:
		return train.optimalscore

#springExperimentConstantPower()
#springExperimentMass()
#powerEffspring()
#springExperiment()
#partialExperiment()
#noNodesExperiment()
if False:
	plotSize = 'huge'
	springExperiment(loadFile = "Experiments/Final/SpringExperimentconfig.csv")
	springExperimentConstantPower( loadFile = "Experiments/Final/SpringExperimentConstantPowerconfig.csv")
	partialExperiment( loadFile = "Experiments/Final/partialExperimentconfig.csv")
	noNodesExperiment( loadFile = "Experiments/Final/No Nodesconfig.csv")
	powerEffspring(loadFile = "Experiments/Final/powereffSpringconfig.csv")
	springExperimentMass(loadFile = "Experiments/Final/SpringExperimentMassconfig.csv")

if False:
	plotSpringExperimentMassNonLinear()
	CMAConvergence(load = True, long = False)
	CMAConvergence(load = True, long = True)
	plotPareto()
	noisySimulation(loadFile = "Experiments/final/noisySimulationconfig.csv")
	paretoCurve(loadFile = "Experiments/Final/paretoCurveconfig.csv")
	
#frameMaker()
#noisySimulation()
noisySimulation(loadFile = "Experiments/final/noisySimulationconfig.csv")
#plotSpringExperimentMassNonLinear()

#longTermExperiment()


#springExperimentMassNonLinear(nonLin = 0)
#springExperimentMassNonLinear(nonLin = 2)
#springExperimentMassNonLinear(nonLin = 10)

#plt.show(block = True)

#springStrengthsExperiment()
#movieMaker()

#paretoCurve()
#paretoCurve(maxOmega = 5)
#paretoCurve(maxOmega = 2)


#CMAConvergence()

#springLengthsExperiment()
#paretoCurve(maxOmega = 20)
#paretoCurve(maxOmega = 1)
