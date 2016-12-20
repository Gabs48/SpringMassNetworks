from collections import deque
import copy
import datetime
import itertools
import numpy as np
import matplotlib
from matplotlib.mlab import *
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
from scipy import signal
from robot import Robot
from utils import *
import sys

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

	def __init__(self, border=1.5, plotCycle=10, startPlot=0, text=False, plot=False, movie=False,\
		pauseTime=0.00001, movieName="out", color=True, delete=True):
		self.plot = plot
		self.movie = movie
		self.color = color
		self.delete = delete
		self.first_it = True
		if plot:
			self.border = border
			self.plotCycle = plotCycle
			self.startPlot = startPlot
		
			self.pauseTime = pauseTime
			self.fig = plt.subplots(figsize=(10.88, 4.88), dpi=300, )
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
					mkdir_p(directory)
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
			plt.ylim(-self.border, self.border + 1.2 * np.max(ypos))
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
					self.text = None#plt.text(0.2,0.9,"",ha='center', va = 'center', transform=plt.gca().transAxes) #Left Align Text
					if self.init:
						self.init = False;
						self.maxy = np.max(ypos)
					plt.ylim(-self.border, self.border + 1.2 * self.maxy)
					if self.first_it == True:
						self.xplotwidth = max(xpos) + self.border *5 - (min(xpos) - self.border)
						self.first_it = False

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
			print "ffmpeg -r " + str(self.fps) + " -s 1100x700"+  " -i "+ self.IMGname + " -c:v libx264 -r 30 -pix_fmt yuv420p " + self.movieName + ".mp4"""
			os.system("ffmpeg -r " + str(self.fps) + " -s 1100x700"+  " -i "+ self.IMGname + " -c:v libx264 -r 30 -pix_fmt yuv420p " +
				 self.movieName + ".mp4""")
			if self.delete:
				for fname in self.fileList:
					os.remove(fname)
			self.fileList = []
			self.frame = 0;

class SimulationEnvironment(object):
	""" class with general Parameters for Simulations but not bound to a specific robot"""
	param = ["timeStep", "simulationLength", "verlet", "refPower", "refDist"]

	def  __init__(self,timeStep=0.005, simulationLength=10000, plot=Plotter(), verlet=True, noisy=False, \
		controlPlot=True, pcaPlot=False, pcaTitle="PCA", pcaFilename="pca", pcaMat=None, perfMetr="dist", \
		refDist=0 , refPower=0):
		self.timeStep = timeStep  # time step size
		self.plot = plot  # plotting
		assert isinstance(simulationLength, int), "simulation length should be integer"
		self.simulationLength = simulationLength # number of iterations
		self.verlet = verlet;
		self.noisy = noisy;
		self.controlPlot = controlPlot
		self.pcaPlot = pcaPlot
		self.pcaTitle = pcaTitle
		self.pcaFilename = pcaFilename
		self.perfMetr = perfMetr
		self.pcaMat = pcaMat
		if self.perfMetr == "powereff" or self.perfMetr == "powersat" or self.perfMetr == "distsat":
			assert refDist is not 0, refPower is not 0
			self.refDist = refDist
			self.refPower = refPower
		else:
			self.refDist = 0
			self.refPower = 0

	def end(self):

		return self.plot.end()

class ControlPlotter(object):
	" plot the generated control signals "

	def __init__(self, robot, simulEnv):
		self.ydata = np.zeros((0, robot.getNoConnections()))
		self.xdata = np.zeros((0,1))
		self.prev_speed = SpaceList(np.zeros(robot.getShape(),float))
		self.simulEnv = simulEnv
		self.timeStep = simulEnv.timeStep
		self.filename = simulEnv.pcaFilename
		self.simulationLength = self.simulEnv.simulationLength
		self.simulationTime = self.simulEnv.timeStep
		self.timeArray = np.linspace(0, self.simulationTime, num=self.simulationLength)
		self.title = simulEnv.pcaTitle
		self.acc =  np.array([])

	def addData(self, robot):
		currentTime = robot.state.currentTime
		restlength = robot.currentRestLength(array = True)
		self.ydata = np.vstack((self.ydata, restlength))
		self.xdata = np.vstack((self.xdata, [[currentTime]]))

	def addPCAData(self, robot):
		"""Store PCA values for plotting limitcycle"""

		speed_it = robot.state.speed
		acc_it = (speed_it.getArray() - self.prev_speed.getArray()) / self.timeStep
		if self.acc.size == 0:
			self.acc = acc_it
		else:
			self.acc = np.vstack((self.acc, acc_it))
		return

	def plot(self):
		fig, ax = Plot.initPlot()
		ax.plot(self.xdata,self.ydata)
		Plot.configurePlot(fig,ax, "time","current SpringLength")
		fig.show()


	def plotLimitCycle(self, n=None, save=True, show=False):
		"""Plot the limit cycle of x_training and y_trained"""

		gap = 2
		pca = None
		window = int(np.ceil(self.acc.shape[0]/gap))

		vec =  self.acc
		if vec.shape[0] > vec.shape[1]:
			if self.simulEnv.pcaMat == None:
				pca = PCA(vec)
				pc1 = pca.Y[:,0]
				pc2 = pca.Y[:,1]
			else:
				pca = self.simulEnv.pcaMat
				res = pca.project(vec)
				pc1 = res[:,0]
				pc2 = res[:,1]
		else:
			pc1 = vec[:,0]
			pc2 = vec[:,1]

		print " -- Save Limit cycle plot in " + self.filename
		fig, ax = Plot.initPlot(proj="3d")
		for j in xrange(1, window):
			ax.plot(pc1[j*gap:(j+1)*gap+1], pc2[j*gap:(j+1)*gap+1], \
				self.timeArray[j*gap:(j+1)*gap+1], \
				c=plt.cm.winter(1.*j/window), linewidth=1.2, label="PCA trajectory")
		ax.set_xlabel('PC 1')
		ax.set_ylabel('PC 2')
		ax.set_zlabel('Time')
		ax.view_init(30, 60)
		plt.title(self.title)
		if show: plt.show()
		if save: plt.savefig(self.filename + ".png", format='png', dpi=300)
		plt.close()

		return pca

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
		self.pcaPlot = simulEnv.pcaPlot
		if self.controlPlot or self.pcaPlot:
			self.controlPlotter = ControlPlotter(robot, simulEnv)

	def simulateStep(self):
		""" Euler integration for a single time step"""
		A = self.robot.computeAcceleration()
		V = self.robot.getVelocity()
		self.iterationNumber+=1
		return self.robot.changeState(self.simulEnv.timeStep, V, A) 

	def runSimulation(self):
		""" Runs a simulation over a number of iterations and returns the distance travelled"""

		for i in range(self.simulEnv.simulationLength):
			self.simulateStep()
			self.simulEnv.plot.update(self.robot,self.iterationNumber)
			if self.controlPlot:
				self.controlPlotter.addData(self.robot)
			if self.pcaPlot:
				self.controlPlotter.addPCAData(self.robot)
		self.simulEnv.end()
		self.endState = self.robot.getState()
		if self.controlPlot:
			self.controlPlotter.plot()
		if self.pcaPlot:
			self.simulEnv.pcaMat = self.controlPlotter.plotLimitCycle()
		return self.performanceMetric()

	def getDistance(self):
		""" Return the current travelled distance """

		return Robot.getDistanceTraveled(self.initState, self.endState)
	
	def performanceMetric(self):
		""" Return a score to characterize the simulation depending on the chosen performance metric """

		# TODOOO: test distsat , powersat and change to speed! (normalized in time)
		distance = self.getDistance()
		speed = distance / (self.iterationNumber * self.simulEnv.timeStep)
		power = self.robot.getPower()
		refPower =self.simulEnv.refPower
		refDist = self.simulEnv.refDist
		C = np.arctanh(1.0 / np.sqrt(2))

		if self.simulEnv.perfMetr == 'dist':
			return [distance, power, distance]
		elif self.simulEnv.perfMetr == 'powereff':
			return [(np.tanh(C * refPower / power) * np.tanh(C * distance / refDist)), power, distance]
		elif self.simulEnv.perfMetr == 'powersat':
			if power > refPower:
				score = (np.tanh(C * refPower / power) * np.tanh(C * distance / refDist))
			else:
				score = (np.tanh(C * distance / refDist))
			return [score, power, distance]
		elif self.simulEnv.perfMetr == 'distsat':
			if distance < refDist:
				score = (np.tanh(C * refPower / power) * np.tanh(C * distance / refDist))
			else:
				score = (np.tanh(C * refPower / power))
			return [score, power, distance]
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
		#print str(self.robot.state.currentTime) + " Updating the states of iteration " + str(self.iterationNumber)
		timeStep = self.simulEnv.timeStep
		self.Aold = self.robot.changeStateVerlet(timeStep, V, self.Aold)
		self.iterationNumber+=1
		#print str(self.robot.state.currentTime) + " Updating iteration number: " + str(self.iterationNumber)
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

class NoisyImpulseVerletSimulation(VerletSimulation):
	""" Simulate noise in Verlet Update steps """
	def __init__(self, simulEnv, robot, noise=1, reset=False, impulserate=0.05, durationRate=100):

		# Parent constructor
		super(NoisyImpulseVerletSimulation, self).__init__(simulEnv, robot, reset=reset)

		# Noise variables
		self.noise = noise
		self.noiseArr = None
		self.durationRate = durationRate
		self.nImpulse = self.simulEnv.simulationLength * impulserate
		self.noiseTime = np.random.randint(0, self.simulEnv.simulationLength, int(np.floor(self.nImpulse)))
		self.noiseIt = 0

	def simulateStep(self):

		self.process()
		V = self.robot.getVelocity()
		timeStep = self.simulEnv.timeStep
		self.iterationNumber+=1

		# If step is a noisy one:
		if self.iterationNumber in self.noiseTime:

			self.noiseIt = 1
			# Estimate impulse noise as percentage of mean acceleration value
			n_val = self.noise * np.mean(np.abs(self.Aold.getArray()))
			x_noise = np.random.uniform(- n_val, n_val)
			y_noise = np.random.uniform(- n_val, n_val)

			# Select a random node
			node_val = np.random.randint(0, self.Aold.getnoNodes())
			self.noiseArr = np.zeros((2, self.Aold.getnoNodes()))
			self.noiseArr[0][node_val] = x_noise
			self.noiseArr[1][node_val] = y_noise
			self.Aold = self.robot.changeStateVerlet(timeStep, V, self.Aold, impulsenoise=self.noiseArr)

		if self.noiseIt > 0:
			if self.noiseIt >= self.durationRate:
				self.noiseIt = 0
			else:
				self.Aold = self.robot.changeStateVerlet(timeStep, V, self.Aold, impulsenoise=self.noiseArr)
				self.noiseIt += 1
				return self.Aold

		if self.noiseIt == 0:
			self.Aold = self.robot.changeStateVerlet(timeStep, V, self.Aold)
			return self.Aold

class TrainingSimulation(VerletSimulation):
	""" Extend VerletSimulation and train an output layer to produce a structured patterns
	The time is divided in five steps as described in the process method.
	Defaut training method use a one-shot regression learning from an input vector formed with the positions
	This class can be extended by rewriting the trainStep, runStep and train methods
	 """

	def __init__(self, simulEnv, robot, omega=5, transPhase=0.2, trainPhase=0.6, trainingPlot="all", \
		signTime=None, outputFilename="sinusoid", outputFolder="RC"):
		""" Init the training test sim class and parent classes
		 - omega is the desired output sinusoid frequency. It should correspond to the frequency of the MSN
		 - transPhase is the proportion of time dedicated to transitoire dynamics before training
		 - trainPhase is the proportion of time dedicated to training
		 - outputFilename and outputFolder are set to save plotting if traingPlot is True
		 - wDistPlot is set to plot the output neurons weight dustribution 
		 - signTime can be set to reduce the phase impact to a significant part of the simulation. For instance, 
		   if the simulation time is set to 50s, the signTime=30, then, the transitoire trinaing and running phases
		   will only applys to 
		"""
		super(TrainingSimulation, self).__init__(simulEnv, robot)

		self.omega = omega
		self.transPhase = transPhase
		self.trainPhase = trainPhase
		self.trainingPlot = trainingPlot
		self.outputFolder = outputFolder
		self.outputFilename = outputFilename

		if not signTime:
			self.signLength = self.simulEnv.simulationLength
		else:
			self.signLength = int(np.floor(signTime / self.simulEnv.timeStep))

		self.simulationTime =  self.simulEnv.timeStep * self.simulEnv.simulationLength
		self.transLength = int(np.floor(self.transPhase * self.signLength))
		self.trainLength = int(np.floor(self.trainPhase * self.signLength))
		self.runLength = self.simulEnv.simulationLength - self.trainLength - self.transLength

		self.inputs = []
		self.yTraining = self.create_y_training()
		self.xTraining = np.array([])
		self.yTrained = np.array([])
		self.nrmsError = None
		self.absError = None
		self.error = None
		self.weightMatrixDiff = None

		self.N = self.robot.getState().pos.getArray().shape[1] + self.robot.getState().speed.getArray().shape[1]
		if len(self.yTraining[self.iterationNumber].shape) != 0:
			self.O = self.yTraining[self.iterationNumber].shape[0]
		else:
			self.O = 1

		if self.robot.control.__class__.__name__ == "ClosedLoopSineControl":
			self.CL = True
			#print " -- Closed-Loop simulation and control activated. Closing the loop from time = " + \
			#	num2str(self.simulEnv.timeStep * (self.transLength + self.trainLength)) + " s -- "
		else:
			self.CL = False

		mkdir_p(self.outputFolder)

	def create_y_training(self):
		""" Create a sinusoid signal of length trainLength to train the output neuron"""

		self.timeArray = np.linspace(0, self.simulationTime, num=self.simulEnv.simulationLength)

		# Single sinus
		# y = np.sin(self.omega * self.timeArray).reshape(-1,1)

		# Real input values

		y = []
		for t in self.timeArray:
			line = connections2Array(self.robot.control.modulationFactorTime(t), self.robot.morph.connections)
			y.append(line)

		return np.array(y)

	def neuron_fct(self, x):
		""" The transition function of the output neuron """

		y =  1 + np.tanh(x)
		return y

	def trainStep(self):
		""" Add training data for a given step """

		posArray = self.robot.getState().pos.getArray().T

		if self.xTraining.size == 0:
			self.xTraining = posArray
		else:
			self.xTraining = np.vstack((self.xTraining, posArray))

	def train(self):
		""" Determine the output weight matrix to minimize error """

		# Copy input, and output
		x = self.xTraining
		y = self.yTraining[self.transLength:self.transLength+self.trainLength]

		# Compute weights
		w, res, rank, singVal = np.linalg.lstsq(x, y)
		np.set_printoptions(threshold=np.inf)
		self.weightMatrix = w
		self.yTrained = np.transpose(np.dot(w.T, x))

		# Print debug
		print " -- Network training by linear regression performed. Sum of residues = {:.4f}".format(res[0]) + \
			". Global NRMSE = {:.4f} --".format(self.nrmse(y, np.dot(x, w)))
		if self.trainingPlot == "all":
			self.plotW()

		# Start Closed-Loop mode
		#self.robot.control.closeLoop()

	def runStep(self):
		""" Run the neuron for a given step """

		# Get input state vector
		if hasattr(self, 'hist'):
			x_it = self.robot.getState().pos.getArray().T
			self.inputs.pop(0)
			self.inputs.append(x_it)
			posArray = np.mat(self.inputs[0])
			for i in range(self.hist-1):
				if i < 3:
					posArray = np.vstack((posArray, self.inputs[i]))
				else:
					if i%4 == 0:
						posArray = np.vstack((posArray, self.inputs[i]))
		else:
			posArray = np.vstack((self.robot.getState().speed.getArray(), self.robot.getState().pos.getArray()))

		# Compute new signal estimation
		y_est = np.asarray(np.transpose(np.dot(self.weightMatrix.T, posArray)))

		# Store estimation in vector
		if self.yTrained.size == 0:
			self.yTrained = y_est
		else:
			self.yTrained = np.vstack((self.yTrained, y_est))

		# Pass the signal estimation to the controller to close the loop
		stepInput = array2ModFactor(y_est, self.robot.morph.connections)
		self.robot.control.setStepInput(stepInput)

	def process(self):
		""" Add the training process to the normal simulation. 5 steps:
		 - Transiant phase: nothing is done
		 - Training phase: adding the data to the training vector
		 - Training point: linear regression to compute the output neurons weight matrix
		 - Runnning point: replace the control inputs by the trained values
		 - End of sim: save and plot """

		it = self.iterationNumber

		# Training phase (add data to trianing vector)
		if it >= self.transLength and it < (self.transLength + self.trainLength - 1):
			print "2. Training phase. It: " + str(it)
			self.trainStep()

		# Training time (linear regression for output neurons)
		if it ==  (self.transLength + self.trainLength - 1):
			#print "3. Training time. It: " + str(it)
			self.train()
			self.runStep()

		# Running phase (connect the neurons output to the robot)
		if it >= (self.transLength + self.trainLength) and it < (self.simulEnv.simulationLength - 1) :
			#print "4. Running phase. It: " + str(it)
			self.runStep()

		# End of simulation (plot everything)
		if it == self.simulEnv.simulationLength - 1:
			#print "5. End of simulation. It: " + str(it)
			self.runStep()
			self.save() ## If save??
			if self.trainingPlot == "all":
				self.plotWDiff(-self.trainLength)
				self.plotError(-self.trainLength+50)
				self.plotInputs()
				self.plot(n=6000)
				self.plotLimitCycle()
			if self.trainingPlot == "cont":
				self.plotWDiff(-self.trainLength*2/3)
				self.plotInputs()
				self.plotError(-self.trainLength+50)
				self.plot(n=6000, comp=2)
				self.plotLimitCycle()

	def mse(self, arr1, arr2):
		""" Compute MSE between two matrices """

		assert arr1.shape == arr2.shape, "Mean Square Error can only be computed on matrices with same size"
		a, b = arr2.shape
		return np.sum((arr2 - arr1) ** 2) / float(a * b)

	def nrmse(self, arr1, arr2):
		""" Compute NRMSE between two matrices """

		# Center signals around 0
		arr1 = arr1 - 1
		arr2 = arr2 - 1
		
		rmse = np.sqrt(self.mse(arr1, arr2))
		max_val = max(np.max(arr1), np.max(arr2))
		min_val = min(np.min(arr1), np.min(arr2))

		return 1 - (rmse / (max_val - min_val))

	def _numPoints(self, n):
		"""Give the number of points to plot in each phase when using n points"""

		# Compute number of points of each phase
		if not n or n > self.simulEnv.simulationLength:
			n = self.simulEnv.simulationLength
		n_tot = self.simulEnv.simulationLength
		n_trans = self.transLength
		n_train = self.trainLength
		n_run = self.runLength
		if n_run > n:
			n_run = n

		return [n_trans, n_train, n_run, n_tot, n]

	def get_training_error(self):
		"""
		Fill and return rms error and absolute for all actuators
		"""

		y_err = []
		if self.nrmsError == None:
			for i in range(self.O):
				y_err.append(self.nrmse(self.yTraining[:,i].reshape(-1,1), self.yTrained[:,i].reshape(-1,1),))

			self.nrmsError = sum(y_err) / float(len(y_err))
			print " -- Computing NRMS Error: " + str(self.nrmsError) + " --"

		y_err = []
		if self.absError == None:
			for i in range(self.O):
				y_norm = np.max(self.yTraining[:,i]) - np.min(self.yTraining[:,i])
				y_err.append(100 * np.max(np.abs(self.yTraining[:,i].reshape(-1,1) - \
					self.yTrained[:,i].reshape(-1,1)) / y_norm))

			self.absError = sum(y_err) / float(len(y_err))
			print " -- Computing Max Absolute Error:  {:.2f} %".format(self.absError) + " --"

		return self.nrmsError, self.absError

	def plot(self, n=None, comp=None, show=False, save=True):
		""" Print the driving and trained signals as a fct of time in a file"""

		[n_trans, n_train, n_run, n_tot, n] = self._numPoints(n)

		# Some arrays init
		nrmse = 0
		y_err = np.zeros(1)

		# Get number of graphs to print
		if comp == None:
			comp = self.O

		for i in range(comp):

			# Compute error vector
			if n_run != 0:
				nrmse = self.nrmse(self.yTraining[-n_run:, i].reshape(-1,1), self.yTrained[-n_run:, i].reshape(-1,1))
				y_norm = np.max(self.yTraining[-n_run:, i]) - np.min(self.yTraining[-n_run:, i])
				y_err = np.abs(self.yTraining[-n_run:, i].reshape(-1,1) - self.yTrained[-n_run:, i].reshape(-1,1)) / y_norm

			# Plot
			print " -- Generating training graph " + str(i+1) + "/" + str(self.yTraining.shape[1]) + ". NRMSE = " + \
				str(nrmse) + " -- "
			fig, ax = Plot.initPlot()
			for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
				item.set_fontsize(17)

			plt.plot(self.timeArray[-n:], self.yTraining[-n:, i] ,"r--", label="Training signal")
			plt.plot(self.timeArray[-n:], np.zeros(n) ,"k-")
			plt.plot(self.timeArray[-n:-n_run-1], self.yTrained[-n:-n_run-1, i] ,"y-", label="Trained sig (train)")
			if n_run != 0:
				plt.plot(self.timeArray[-n_run-1:], self.yTrained[-n_run-1:, i] ,"b-", label="Trained sig (run)")
				#plt.plot(self.timeArray[-n_run:], y_err[-n_run:] ,"g-", label="Error signal")

			plt.title("Spring control force " + str(i+1) + ". Maximum error =  {:.2f} %".format(100 * np.max(y_err)) + \
				". NRMSE = {:.2f}".format(nrmse))
			Plot.configurePlot(fig, ax, 'Temp', 'Temp', legend=False, legendLocation='lower center')
			plt.xlabel('Simulation step')
			plt.ylabel('Signal value')
			if show: plt.show()
			if save: plt.savefig(self.outputFolder + "/" + self.outputFilename + "_sin_" + str(i+1) + ".png", format='png', dpi=300)
			plt.close()

	def plotInputs(self):
		""" Plot inputs evolution to monitor feddback effect """

		plt.plot(self.timeArray, self.acc, "b-")
		plt.title("Node 1 acceleration evolution")
		plt.savefig(self.outputFolder + "/" + self.outputFilename + "_acc.png", format='png', dpi=300)
		plt.close()

		plt.plot(self.timeArray, self.pos, "")
		plt.title("Node 1 position evolution")
		plt.savefig(self.outputFolder + "/" + self.outputFilename + "_pos.png", format='png', dpi=300)
		plt.close()

		plt.plot(self.timeArray, self.speed, "")
		plt.title("Node 1 speed evolution")
		plt.savefig(self.outputFolder + "/" + self.outputFilename + "_speed.png", format='png', dpi=300)
		plt.close()

	def plotW(self, show=False, save=True):
		""" Plot distribution of wieight matrix W """

		fig, ax = Plot.initPlot()
		plt.hist(self.weightMatrix.reshape(-1, 1), bins=40)
		plt.title("Distribution of output layer weight matrix")
		plt.xlabel("Value")
		plt.ylabel("Frequency")
		Plot.configurePlot(fig, ax, 'Temp', 'Temp', legend=False)
		if show: plt.show()
		if save: plt.savefig(self.outputFolder + "/" + self.outputFilename + "_w_hist.png", format='png', dpi=300)
		plt.close()

	def save(self, filename=None):
		""" Save the weights matrix for further simulations """

		if filename == None:
			filename = "weight_matrix_" + timestamp() + ".pkl"

		#dump_pickle(self, filename)

	def plotWDiff(self, n=None, show=False, save=True):
		""" Plot evolution of the weight matrix differences """
		
		neg = False
		if n != None: 
			if n < 0:
				neg = True
				n = abs(n)

		[n_trans, n_train, n_run, n_tot, n] = self._numPoints(n)
		if n > n_train - 2:
			n = n_train - 2

		fig, ax = Plot.initPlot()
		if neg:
			plt.plot(self.timeArray[n_tot-n_run-n:n_tot-n_run], self.weightMatrixDiff[-n:])
		else:
			plt.plot(self.timeArray[n_trans:n_trans+n], self.weightMatrixDiff[0:n])
		plt.title("Evolution of the trained weights with time")
		plt.xlabel("Time")
		plt.ylabel("Weight matrix derivative")
		Plot.configurePlot(fig, ax, 'Temp', 'Temp', legend=False)
		if show: plt.show()
		if save: plt.savefig(self.outputFolder + "/" + self.outputFilename + "_w_diff.png", format='png', dpi=300)

	def plotError(self, n=None, show=False, save=True):
		""" Plot error evolution"""
		
		neg = False
		if n != None: 
			if n < 0:
				neg = True
				n = abs(n)

		[n_trans, n_train, n_run, n_tot, n] = self._numPoints(n)
		if n > n_train - 2:
			n = n_train - 2

		fig, ax = Plot.initPlot()
		if neg:
			plt.plot(self.timeArray[n_tot-n_run-n:n_tot-n_run], self.error[-n:])
		else:
			plt.plot(self.timeArray[n_trans:n_trans+n], self.error[0:n])
		plt.title("Error evolution")
		plt.xlabel("Time")
		plt.ylabel("Error")
		if show: plt.show()
		if save: plt.savefig(self.outputFolder + "/" + self.outputFilename + "_error.png", format='png', dpi=300)

	def plotLimitCycle(self, n=None, show=False, save=True):
		"""Plot the limit cycle of x_training and y_trained"""

		[n_trans, n_train, n_run, n_tot, n] = self._numPoints(n)
		gap = 2
		window = int(np.ceil((n-n_run-n_trans-1)/gap))

		vec = self.xTraining.T
		if vec.shape[0] > vec.shape[1]:
			res = PCA(vec)
			pc1 = res.Y[:,0]
			pc2 = res.Y[:,1]
		else:
			pc1 = vec[:,0]
			pc2 = vec[:,1]

		fig, ax = Plot.initPlot()#proj="3d")
		for j in xrange(1, window):
			ax.plot(pc1[j*gap:(j+1)*gap+1], pc2[j*gap:(j+1)*gap+1], \
				#self.timeArray[n_trans+1+j*gap:n_trans+1+(j+1)*gap+1], \
				c=plt.cm.winter(1.*j/window), linewidth=1.2, label="PCA trajectory")
		ax.set_xlabel('PC 1')
		ax.set_ylabel('PC 2')
		#ax.set_zlabel('Time')
		#ax.view_init(30, 60)
		if show: plt.show()
		if save: plt.savefig(self.outputFolder + "/" + self.outputFilename + "_lc.png", format='png', dpi=300)
		plt.close()

		return

class ForceTrainingSimulation(TrainingSimulation):
	""" Extend the TrainingSimulation class to use FORCE online learning """

	## Parameters to discuss:
	# Beta and alpha
	# closing the loop completely
	# open loop previous to training
	# Memory of previous accelerations
	def __init__(self, simulEnv, robot, transPhase=0.2, trainPhase=0.6, trainingPlot=True, \
		alpha=1, beta=0.1, openPhase=0.1, signTime=None, wDistPlot=True, outputFilename="sinusoid", \
		outputFolder="RC", printPhase=True):
		""" Init class: phases are reparted like this
		- Transition phase: nothing happens here, we let the dynamics stabilizes
		- Training phase: here we do online weights value training. The trianing phase itself is divided in three parts:
			- OpenLoop phase: here, we start training but we won't feed the generated signal until it stabilizes
			- Closing phase: here, we gradually mix the feedback signal sith the target signal
		- The rest is dedicates to Run phase
		 """

		# Fix here the training and running phase if needed
		super(ForceTrainingSimulation, self).__init__(simulEnv, robot, transPhase=transPhase,  \
			trainPhase=trainPhase, trainingPlot=trainingPlot, \
			signTime=signTime, outputFilename=outputFilename, outputFolder=outputFolder)

		# Class variables
		self.openPhase = openPhase
		self.openLength = int(np.floor(self.openPhase * self.trainLength))
		self.closedPhase = 1 - self.openPhase
		self.closedLength =  int(np.floor(self.closedPhase * self.trainLength))
		self.printPhase = printPhase

		# Algorithm constants
		self.alpha = alpha
		self.beta = beta
		self.hist = 3
		self.eWin = 5

		# Algorithm matrices
		self.trainIt = 0
		self.a = 0
		self.po = 0
		self.v = 0
		self.w = None
		self.p = None
		self.error = None
		self.yTrained = np.array([])
		self.acc = np.array([])
		self.accf = np.array([])
		self.speed = np.array([])
		self.pos = np.array([])

		# Create actuator threshold and low pass filtering
		self.fc = 3 # Hz
		## REPLACE THOSE ABSOLUTE VALUES
		self.thh = 1 + 0.5  # Higher value of thresholding
		self.thl = 1 - 0.5 # Lower value for thresholding
		self.order = 120 # Filter order
		self.buffLen = 3 * self.order # Signal buffer length for filtering
		self.filt_b = signal.firwin(self.order, self.fc * self.simulEnv.timeStep)
		self.filt_a = [1]
		#signal.butter(self.order, self.fc * self.simulEnv.timeStep, 'low')
		self.filt_fifo = [] # Create a fifo
		self.filt_fifo_2 = [] # Create a fifo

	def createNeuronLayer(self, size, x):
		""" Create the weights values for an intermediate layer of neurons"""

		self.neuron_layer_size = size
		self.neuron_layer_norm_factor = 10
		self.neuron_layer_w = []
		self.neuron_layer_b = []

		for i in range(size):
			self.neuron_layer_w.append(2 * np.array(np.random.rand(x.shape[0])) - 1)
			self.neuron_layer_b.append(0.5 * np.random.rand() - 0.25)

	def neuronLayerFct(self, x):
		""" Run non-linear function on input vector through an neuronal layer"""

		y = []
		for n in range(self.neuron_layer_size):
			x = x / self.neuron_layer_norm_factor
			y.append(np.tanh(np.dot(self.neuron_layer_w[n], x)))# + self.neuron_layer_b[n]))

		return np.mat(y).T

	def physActFilter(self, predSig):
		"""
		Apply filtering and thresholding to model physical actuation properties and 
		avoid numerical instabilities
		"""
		# Threshold current value
		#print "Predicted signal: " + str(predSig)
		thSig = np.clip(predSig, self.thl, self.thh)
		#print "Thresholded signal: " + str(thSig)

		# Get prev time_steps and convert to numpy matrix
		if not self.filt_fifo:
			for i in range(self.buffLen):
				self.filt_fifo.append(np.zeros(predSig.shape))
		self.filt_fifo.pop(0)
		self.filt_fifo.append(thSig)
		sigMat = np.mat(self.filt_fifo[0])
		for i in range(len(self.filt_fifo)):
			sigMat = np.vstack((sigMat, self.filt_fifo[i]))

		# Filter (TODO: filtering too much useless points each timestep here)
		filtSig = np.zeros(sigMat.shape[1])
		for i in range(sigMat.shape[1]):
			filtSig[i] = signal.lfilter(self.filt_b, self.filt_a, sigMat[:, i].T)[:, -1]

		return filtSig

	def physSensFilter(self, sig):
		"""
		Apply filtering and thresholding to model physical sensor properties and 
		avoid numerical instabilities
		"""

		# Get prev time_steps and convert to numpy matrix
		if not self.filt_fifo_2:
			for i in range(self.buffLen):
				self.filt_fifo_2.append(np.zeros(sig.shape))
		self.filt_fifo_2.pop(0)
		self.filt_fifo_2.append(sig)
		sigMat = np.mat(self.filt_fifo_2[0])
		for i in range(len(self.filt_fifo_2)):
			sigMat = np.vstack((sigMat, self.filt_fifo_2[i]))


		# Filter (TODO: filtering too much useless points each timestep here)
		filtSig = np.zeros(sigMat.shape[1])
		for i in range(sigMat.shape[1]):
			filtSig[i] = signal.lfilter(self.filt_b, self.filt_a, sigMat[:, i].T)[:, -1]

		return filtSig

	def runStep(self):
		""" Run the neuron for a given step """

		# Get robot current state
		a_it = self.Aold.getArray()[0, :]
		v_it = self.robot.getState().speed.getArray()[0, :]
		pos_it = self.robot.getState().pos.getArray()[0, :]
		da_it = a_it - self.a
		self.a = a_it
		self.po = pos_it
		self.v = v_it

		# Process acceleration vector through a non-linear layer
		x_it = np.mat(self.physSensFilter(a_it)).T #self.neuronLayerFct(pos_it)

		# Get input vector
		if not self.inputs:
			raise('Error: No trianing before running')
		self.inputs.pop(0)
		self.inputs.append(x_it)

		x = np.mat(self.inputs[0])
		for i in range(self.hist-1):
			#if i < 4:
			x = np.vstack((x, self.inputs[i]))
			#else:
			#if i%4 == 0:
			#	x = np.vstack((x, self.inputs[i]))

		# Compute new estimation and filter it
		y_est = self.neuron_fct(np.asarray(self.w_prev.T * x).T)
		filt_y_est = y_est[0]

		# Store estimation in vector
		if self.yTrained.size == 0:
			self.yTrained = filt_y_est
		else:
			self.yTrained = np.vstack((self.yTrained, filt_y_est))
		self.acc = np.vstack((self.acc, x[0:1].T))
		self.speed = np.vstack((self.speed, v_it))
		self.pos = np.vstack((self.pos, pos_it))

		# Filter and pass the signal estimation to the controller to close the loop
		
		self.robot.control.setStepInput(filt_y_est)

		# Print
		if self.iterationNumber == self.simulEnv.simulationLength - 1:
			if self.printPhase:
				self.printSim()

	def trainStep(self):
		""" Add training data for a given step """

		# Get robot current state
		#print str(self.robot.state.currentTime) + " Getting states of iteration " + str(self.iterationNumber)
		a_it = self.Aold.getArray()[0, :]
		v_it = self.robot.getState().speed.getArray()[0, :]
		pos_it = self.robot.getState().pos.getArray()[0, :]
		da_it = a_it - self.a
		self.a = a_it
		self.po = pos_it
		self.v = v_it

		# Create a non-linear layer of neurons and process acc vector through it
		#if self.trainIt == 0:
		#	self.createNeuronLayer(20, pos_it)
		x_it = np.mat(self.physSensFilter(a_it)).T # self.neuronLayerFct(pos_it)

		# If the inputs fifo hasen't been created, do it
		if not self.inputs:
			self.N = x_it.shape[0]
			for i in range(self.hist):
				self.inputs.append(np.zeros((self.N, 1)))

		# Update the inputs fifo (TODO: useless if the whole xtraining is already savec below)
		self.inputs.pop(0)
		self.inputs.append(x_it)

		# Fill the xTraining vector (usefull for plotting limit cycle)
		if self.xTraining.size == 0:
			self.xTraining = x_it
		else:
			self.xTraining = np.hstack((self.xTraining, x_it))
		
		# Get current learning algo inputs and supervized ouput
		x = np.mat(self.inputs[0])
		for i in range(self.hist-1):
			#if i < 4:
			x = np.vstack((x, self.inputs[i]))
			#else:
			#if i%4 == 0:
			#	x = np.vstack((x, self.inputs[i]))

		y = np.mat(self.yTraining[self.iterationNumber+1])

		# If first iteration, init with random weights
		if self.trainIt == 0:
			w = np.random.rand(x.shape[0], self.O)
			p = np.identity(x.shape[0]) / self.alpha
			yTrained = self.neuron_fct(np.asarray(w.T * x).T)
			filtYTrained = yTrained[0]
			self.yTrained = filtYTrained

			self.acc = a_it[0]
			self.speed = v_it
			self.pos = pos_it
			#self.error = y - yTrained

		# Else update
		else:

			# Update inverse Correlation Matrix of the network states
			p_prev = np.mat(self.p_prev)
			den = 1 + x.T * p_prev * x
			num = p_prev * x * x.T * p_prev
			p = p_prev - num / den

			# Update weight matrix
			w_prev = np.mat(self.w_prev)
			
			# Compute minimal error a window
			if self.iterationNumber > self.eWin:
				e_lim_min = self.eWin
			else:
				e_lim_min = self.iterationNumber
			if self.iterationNumber < self.simulEnv.simulationLength - self.eWin:
				e_lim_max = self.eWin
			else:
				e_lim_max = self.simulEnv.simulationLength - self.iterationNumber
			e_i = self.neuron_fct(w_prev.T * x) - np.mat(self.yTraining[self.iterationNumber-e_lim_min]).T
			for i in range(self.iterationNumber-e_lim_min+1, self.iterationNumber+e_lim_max):
				e_i = np.hstack((e_i, self.neuron_fct(w_prev.T * x) - np.mat(self.yTraining[i]).T))
			e_p = np.amin(e_i, axis=1)
			#e_p = w_prev.T * x - y.T
			#e_p = e_i[:, e_p_arg]

			# Update weight matrix
			l = (p * x) / (1 + x.T * p * x)
			w = w_prev - p * x * e_p.T

			# Fill the w error vector (usefull for plotting W convergence)
			if self.weightMatrixDiff == None:
				self.weightMatrixDiff = np.max(np.abs(w - w_prev))
			else:
				self.weightMatrixDiff = np.hstack((self.weightMatrixDiff, np.max(np.abs(w - w_prev))))

			# Update output
			yTrained =  self.neuron_fct(np.asarray(w.T * x).T)
			filtYTrained = yTrained[0]
			self.yTrained = np.vstack((self.yTrained, np.asarray(filtYTrained)))
			self.acc = np.vstack((self.acc, a_it[0]))
			self.speed = np.vstack((self.speed, v_it))
			self.pos = np.vstack((self.pos, pos_it))

			# Update error (usefull for plotting error evolution)
			if self.error == None:
				self.error = np.mean(np.abs(y - filtYTrained))
			else:
				self.error = np.hstack((self.error, np.mean(np.abs(y - filtYTrained))))

		# start Closed-Loop mode
		if self.trainIt == self.openLength:
			self.robot.control.closeLoop(self.closedLength, beta=self.beta)

		# Pass the signal estimation to the controller to close the loop
		self.robot.control.setStepInput(filtYTrained)

		# Update iteration
		self.trainIt += 1
		self.w_prev = w
		self.p_prev = p

		return

	def printSim(self):
		""" Save some useful information regarding the simulation proceeding """

		with open("./" + self.outputFolder + "/" + self.outputFilename + ".txt", "w+") as file:
			file.write("  Phase name   |  length  |  t_start |  t_stop |\n")
			file.write("------------------------------------------------\n")
			file.write("  Transitoire  |   " + str(self.transLength) + "   |   " + \
				str(0.0) + "    |   " + \
				str(self.transLength*self.simulEnv.timeStep) + "  |\n")
			file.write("  Training     |   " + str(self.trainLength) + "   |   " + \
				str(self.transLength*self.simulEnv.timeStep) + "   |   " + \
				str((self.trainLength+self.transLength)*self.simulEnv.timeStep) + "  |\n")
			file.write("  Running      |   " + str(self.runLength) + "   |   " + \
				str((self.trainLength+self.transLength)*self.simulEnv.timeStep) + "   |   " + \
				str(self.simulEnv.simulationLength*self.simulEnv.timeStep) + "  |\n")
			file.write("------------------------------------------------\n")

			file.write("  Open Loop    |   " + str((self.openLength)) + "   |   " + \
				str(self.transLength*self.simulEnv.timeStep) + "   |   " + \
				str((self.transLength+self.openLength)*self.simulEnv.timeStep) + "  |\n")
			file.write("  Closing Loop |   " + str(self.closedLength) + "   |   " + \
				str((self.transLength+self.openLength)*self.simulEnv.timeStep) + "   |   " + \
				str((self.transLength+self.openLength+self.closedLength)*self.simulEnv.timeStep) + "  |\n")
			file.write("------------------------------------------------\n")
			file.write("Parameters: alpha=" + str(self.alpha) + "     beta=" + str(self.beta))
			file.close()

	def train(self):
		""" Nothing to do here as FORCE is an online method """

		self.weightMatrix = self.w_prev
		if self.trainingPlot == "cont" or self.trainingPlot == "all" :
			self.plotW()

		# Start Closed-Loop mode
		print " -- Training phase finished -- "
		#self.robot.control.closeLoop()

		return

class TrainedSimulation(VerletSimulation):
	""" Run a simulation of a previously trained class based on the training matrix"""

	def __init__(self, simulEnv, robot, filename=None, reset=False, transPhase=0.5):
		""" Init class """

		super(TrainedSimulation, self).__init__(simulEnv, robot, reset=reset)

		self.filename = filename

		self.N = 0
		self.fifo_size = None
		self.x_fifo = []
		self.y_hist = np.array([])
		self.weightMatrix = None
		self._get_data()

		self.transPhase = transPhase
		self.simulationTime =  self.simulEnv.timeStep * self.simulEnv.simulationLength
		self.transLength = int(np.floor(self.transPhase * self.simulEnv.simulationLength))

	def _get_data(self):
		""" Fill the weight matrix given a file """

		sim = load_pickle(self.filename)
		self.N = copy.copy(sim.weightMatrix)
		self.fifo_size = copy.copy(sim.hist)
		self.weightMatrix = copy.copy(sim.weightMatrix)

	def process(self):
		""" Compute driving signals and input them to the robot """

		# Get current acceleration
		x_it = self.Aold.getArray().T

		# Fill a FIFO with accelerations
		if not self.x_fifo:
			self.N = x_it.shape[0]
			for i in range(self.fifo_size):
				self.x_fifo.append(np.zeros((self.N, 1)))
		self.x_fifo.pop(0)
		self.x_fifo.append(x_it)

		# From FIFO of accelerations, construct x vector
		x = np.mat(self.x_fifo[0])
		for i in range(self.fifo_size-1):
			x = np.vstack((x, self.x_fifo[i]))

		# Compute new estimation
		y_est = np.asarray(self.weightMatrix.T * x).T

		# Store estimation in vector
		if self.y_hist.size == 0:
			self.y_hist = y_est
		else:
			self.y_hist = np.vstack((self.y_hist, y_est))

		# Pass the signal estimation to the controller to close the loop
		stepInput = array2ModFactor(y_est, self.robot.morph.connections)
		self.robot.control.setStepInput(stepInput)

		print "Distance: " + str(self.robot.state.pos.getArray()[0, 0])
		
		# start Closed-Loop mode
		if self.iterationNumber  > self.transLength:
			fig, ax = Plot.initPlot()#proj="3d")
			ax.plot(self.y_hist[:, 0])
			plt.savefig("a.png", format='png', dpi=300)
			self.robot.control.closeLoop(0, beta=1)

#class TrainedNoisySimulation(TrainedSimulation):
# Inherit from two class?
	""" Run a simulation of a previously trained class based on the training matrix"""