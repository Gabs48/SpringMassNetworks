import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
import itertools
from robot import Robot
from utils import SpaceList, Plot, LinearMap, NeuralNetwork, num2str

class Plotter(object):
	""" create instance with plotting properties 
		--- attributes---
		- border: float
			space at the border of the plot diagram
		- plotCycle : int
			number of iterations before the plot is refreshed
		- startPlot : int
			iteration number at which plotting starts
		-  plot: boolean
			should a plot be shown
		- pauseTime : float
			time that is paused on the plot
		"""

	def __init__(self, border=1.5, plotCycle=10, startPlot=0, text = True, plot = False, movie = False, pauseTime = 0.00001, movieName = "out", color = True):
		self.plot = plot
		self.movie = movie
		self.color = color
		if plot:
			self.border = border
			self.plotCycle = plotCycle
			self.startPlot = startPlot
		
			self.pauseTime = pauseTime
			self.fig = plt.subplots(figsize=(10.88, 6.88), dpi=100, )
			if text:
				self.text = plt.text(0.2,0.9,"",ha='center', va = 'center', transform=plt.gca().transAxes)#Left Align Text
			else:
				self.text = None
			self.init = True # plot in initial modus
			xlist=[];
			ylist=[];
			self.plt_line, = plt.plot(xlist, ylist)
			# init drawing
			plt.fill_between([-1e8, 1e8], -20, 0, 
				facecolor='gray', edgecolor='gray')
			if movie:
				self.fileList = []
				self.fps = 30
				self.frame = 0
				if os.name == "posix":
					directory = os.getenv("HOME") + '/.temp'
					os.mkdir(directory)
					self.IMGname =  directory + '_tmp%04d.png'
				else:
					self.IMGname = 'temp/_tmp%04d.png'
				self.movieName = movieName
			if color:
				jet = cm = plt.get_cmap('spectral') 
				cNorm  = colors.Normalize(vmin = 0.5, vmax = 1.5)
				self.colorMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

	def _construct_plot_lines(self, xpos, ypos, connections):
		""" update properties of instance based on x and y coordinates and connections matrix"""
		if self.init:
			self.init = False;
			plt.ylim(-self.border, self.border+ 3 * np.max(ypos))
			self.xplotwidth = max(xpos)+self.border - (min(xpos)-self.border)
		
		"""all_lines = np.dstack((np.tile(xpos[:, np.newaxis], (1, len(xpos))),
								np.tile(ypos[np.newaxis, :], (len(ypos), 1))))
		all_lines = np.reshape(all_lines, (-1, 2))"""

		xlist = []
		ylist = []
		for i,j in itertools.product(range(len(xpos)), range(len(ypos))):
			if connections[i,j]:
				xlist.append(xpos[i])
				xlist.append(xpos[j])
				xlist.append(None)
				ylist.append(ypos[i])
				ylist.append(ypos[j])
				ylist.append(None)
				
		return xlist, ylist

	def update(self, robot, iterationCount = 0):
		self.draw(robot, iterationCount)

	def draw(self, robot, iterationCount):
		""" draw a plot of the robot parameters"""
		# draw robot
		if self.plot:
			if iterationCount % self.plotCycle == self.startPlot % self.plotCycle:
				xpos, ypos, connections = robot._getRobotPos2D()
				if self.color:
					stressRatio = robot.stressRatio()
					plt.cla()
					plt.fill_between([-1e8, 1e8], -20, 0, facecolor='gray', edgecolor='gray')
					self.text = plt.text(0.2,0.9,"",ha='center', va = 'center', transform=plt.gca().transAxes) #Left Align Text
					if self.init:
						self.init = False;
						self.maxy = np.max(ypos)
					plt.ylim(-self.border, self.border + 3 * self.maxy)
					self.xplotwidth = max(xpos) + self.border - (min(xpos) - self.border)

					for i,j in itertools.product(range(len(xpos)), range(len(ypos))):
						if connections[i,j]:
							colorVal = self.colorMap.to_rgba(stressRatio[i,j])
							plt.plot([xpos[i], xpos[j]], [ypos[i], ypos[j]], color = colorVal)
					plt.plot(xpos, ypos, 'ko', markersize = 10)

				else:
					xlist, ylist = self._construct_plot_lines(xpos, ypos, connections)
					self.plt_line.set_xdata(xlist)
					self.plt_line.set_ydata(ylist)
				minx = min(xpos)-self.border
				plt.xlim(minx, minx + self.xplotwidth)

				if self.text is not None:
					self.text.set_text(robot.printState())
				if not self.movie:
					plt.draw()
				else:
					self.frame+= 1;
					fname = self.IMGname % self.frame
					plt.savefig(fname)
					self.fileList.append(fname)
				plt.pause(self.pauseTime)

	def end(self):
		if self.movie:
			try:
				os.remove(self.movieName + ".mp4")
			except OSError:
				pass
			os.system("ffmpeg -r " + str(self.fps) + " -s 1100x700"+  " -i "+ self.IMGname + " -c:v libx264 -r 30 -pix_fmt yuv420p " +
				 self.movieName + ".mp4""")
			for fname in self.fileList:
				os.remove(fname)
			self.fileList = []
			self.frame = 0;

class SimulationEnvironment(object):
	""" class with general Parameters for Simulations but not bound to a specific robot"""
	param = ["timeStep", "simulationLength", "verlet"]
	def  __init__(self,timeStep = 0.005, simulationLength=10000, plot = Plotter(), verlet = True, controlPlot = True, 
				perfMetr = "dist", refDist = 0, refPower = 0):
		self.timeStep = timeStep  # time step size
		self.plot = plot  # plotting
		assert isinstance(simulationLength, int), "simulation length should be integer"
		self.simulationLength = simulationLength # number of iterations
		self.verlet = True;
		self.controlPlot = controlPlot
		self.perfMetr = perfMetr # either "dist" or "powereff"
		if self.perfMetr == "powereff":
			assert refDist is not 0, refPower is not 0
			self.refDist = refDist
			self.refPower = refPower

	def end(self):
		return self.plot.end()

class ControlPlotter(object):
	" plot the generated control signals "
	def __init__(self, robot):
		self.ydata = np.zeros((0, robot.getNoConnections()))
		self.xdata = np.zeros((0,1))

	def addData(self, robot):
		currentTime = robot.state.currentTime
		restlength = robot.currentRestLength(array = True)
		self.ydata = np.vstack((self.ydata, restlength))
		self.xdata = np.vstack((self.xdata, [[currentTime]]))

	def plot(self):
		fig, ax = Plot.initPlot()
		ax.plot(self.xdata,self.ydata)
		Plot.configurePlot(fig,ax, "time","current SpringLength")
		fig.show()

class Simulation(object):
	""" class to run and store simulation runs, for better results the 
		child class VerletSimulation is advised """
	param = ["robot", "simulEnv", "initState"]
	def __init__(self,simulEnv ,robot):
		self.simulEnv = simulEnv
		self.robot = robot
		self.initState = robot.getState()
		self.endState = None
		self.iterationNumber = 0;
		self.controlPlot = simulEnv.controlPlot
		if self.controlPlot:
			self.controlPlotter = ControlPlotter(robot)

	def simulateStep(self):
		""" Euler integration for a single time step"""
		A = self.robot.computeAcceleration()
		V = self.robot.getVelocity()
		timeStep = self.simulEnv.timeStep
		self.iterationNumber+=1
		return self.robot.changeState(timeStep, V, A) 

	def runSimulation(self):
		""" Runs a simulation over a number of iterations and returns the distance travelled"""

		for i in range(self.simulEnv.simulationLength):
			self.simulateStep()
			self.simulEnv.plot.update(self.robot,self.iterationNumber)
			if self.controlPlot:
				self.controlPlotter.addData(self.robot)
		self.simulEnv.end()
		self.endState = self.robot.getState()
		if self.controlPlot:
			self.controlPlotter.plot()
		return self.performanceMetric()
	
	def performanceMetric(self):
		distance = Robot.getDistanceTraveled(self.initState, self.endState)
		if self.simulEnv.perfMetr == 'dist':
			return distance
		elif self.simulEnv.perfMetr == 'powereff':
			power = self.robot.getPower()
			refPower =self.simulEnv.refPower
			refDist = self.simulEnv.refDist
			C = np.arctanh(1.0 / np.sqrt(2))
			return (np.tanh(C * refPower / power) * np.tanh(C * distance / refDist))
		else:
			raise NotImplementedError ('the requested performance metric has not been implemented')

class VerletSimulation(Simulation):
	""" use the Verlet algorithm to obtain more accurate simulations """
	def __init__(self, simulEnv, robot, reset = False):
		if reset: robot.reset()
		super(VerletSimulation, self).__init__(simulEnv, robot)
		self.Aold = SpaceList(np.zeros(robot.getShape(),float))


	def process(self):
		""" extend simulateStep in child classes """
		
	def simulateStep(self):
		self.process()
		V = self.robot.getVelocity()
		timeStep = self.simulEnv.timeStep
		self.iterationNumber+=1
		self.Aold = self.robot.changeStateVerlet(timeStep, V, self.Aold)
		return self.Aold

	def getTime(self):
		return self.robot.state.currentTime

class NoisyVerletSimulation(VerletSimulation):
	""" Simulate noise in Verlet Update steps """
	def __init__(self, simulEnv, robot, noise = 0.01, reset = False):
		self.noise = noise
		super(NoisyVerletSimulation, self).__init__(simulEnv, robot, reset = reset)

	def simulateStep(self):
		self.process()
		V = self.robot.getVelocity()
		timeStep = self.simulEnv.timeStep
		self.iterationNumber+=1
		self.Aold = self.robot.changeStateVerlet(timeStep, V, self.Aold, self.noise)
		return self.Aold

class ReservoirSimulation (VerletSimulation):
	""" perform reservoir computing on MSNs """
	def __init__(self, simulEnv, robot, connArray, sampleInterval, targetFunction, relNoise, mapping = LinearMap(), delayTime = 0, delayStep = 1):
		super(ReservoirSimulation, self).__init__(simulEnv, robot)
		self.mapping = mapping
		self.connArray = connArray
		self.dimReservoir = len(connArray)
		self.delayTime = delayTime
		self.delayStep = delayStep

		for connectionNumber in connArray:
			assert connectionNumber < robot.getNoConnections()
		self.sampleInterval = sampleInterval
		self.targetFunction = targetFunction
		self.relNoise = relNoise
		try:
			self.dimTarget = len(targetFunction(robot))
		except:
			self.dimTarget = 1
		self.sampleNum =  int(self.simulEnv.simulationLength / self.sampleInterval )

		self.reservoirArray = np.zeros((self.dimReservoir, self.sampleNum))
		self.targetArray = np.zeros((self.dimTarget, self.sampleNum))

	def process(self):
		if (self.iterationNumber % self.sampleInterval == 0):
			countSample = int(self.iterationNumber / self.sampleInterval)
			# read reservoir values
			for i in range(self.dimReservoir):
				connectionNumber = self.connArray[i]
				noise  = 1
				if self.relNoise > 0:
					noise = np.random.normal(1, self.relNoise)
				self.reservoirArray[i, countSample] = self.robot.getDistance(connectionNumber) * noise
			# determine target values
			self.targetArray[:, countSample] = self.targetFunction(self.robot)

	def map(self, trainFraction = 0.5):

		# add delayline data 
		effNoSamples = self.sampleNum - self.delayTime * self.delayStep
		input = np.zeros(( 0, effNoSamples ))
		target = self.targetArray[:, self.delayTime * self.delayStep : ]
		for i in range(self.delayTime + 1):
			input = np.vstack((input, self.reservoirArray[:, i * self.delayStep : effNoSamples + i * self.delayStep]))

		self.trainSampleNum = int(effNoSamples * trainFraction)
		self.testSampleNum = effNoSamples- self.trainSampleNum

		# Training (mapping can be a NeuralNetwork or a LinearMap instance)
		self.trainSamples = input[:, 0 : self.testSampleNum]
		self.trainTarget = target[:, 0 : self.testSampleNum]
		self.weights, self.trainOutput, self.trainError, self.trainRelError = self.mapping.train(self.trainSamples, self.trainTarget)
		
		# Testing
		self.testSamples = input[:, self.testSampleNum :]
		self.testTarget = target[:, self.testSampleNum :]
		self.testOutput, self.testError, self.testRelError = self.mapping.test( self.testSamples, self.testTarget)

	def performanceMetric(self):
		self.map()
		return self.testError

	def plot(self, targetnum = 0, block = False):
		noTrain = len(self.trainTarget[targetnum])
		noTest = len(self.testTarget[targetnum])

		x1= np.arange(noTrain) * self.simulEnv.timeStep
		x2 = (noTrain + np.arange(noTest)) * self.simulEnv.timeStep

		fig, ax = Plot.initPlot()
		ax.set_title(r'$ \epsilon_{train} = %(num1)s \ \ \ \epsilon_{test} = %(num2)s $' % {'num1': num2str(self.trainRelError), 'num2': num2str(self.testRelError)}, fontsize = 30)
		ax.plot(np.hstack((x1,x2)), np.hstack((self.trainTarget[targetnum], self.testTarget[targetnum])), label = "target", linewidth = 3, alpha = 0.5)
		ax.plot(x1, self.trainOutput[targetnum], 'k-', label = "train")
		ax.plot(x2, self.testOutput[targetnum], 'r.', label = "test")
		ax.legend()
		Plot.configurePlot(fig,ax,"time","value", legendLocation = 'lower center')
		plt.show(block = block)

		return fig, ax

