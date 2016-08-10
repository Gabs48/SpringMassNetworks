import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
import itertools
from robot import Robot
from utils import SpaceList, Plot, LinearMap, NeuralNetwork, num2str, mkdir_p, connections2Array, array2Connections

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

	def getDistance(self):
		""" Return the current travelled distance """

		return Robot.getDistanceTraveled(self.initState, self.endState)
	
	def performanceMetric(self):
		""" Return a score to characterize the simulation depending on the chosen performance metric """

		distance = self.getDistance()
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

class TrainingSimulation(VerletSimulation):
	""" Extend VerletSimulation and train an output layer to produce a structured patterns
	The time is divided in five steps as described in the process method.
	Defaut training method use a one-shot regression learning from an input vector formed with the positions
	This class can be extended by rewriting the trainStep, runStep and train methods
	 """

	def __init__(self, simulEnv, robot, omega=5, transPhase=0.2, trainPhase=0.6, trainingPlot=True, \
		signTime=None, wDistPlot=True, outputFilename="sinusoid", outputFolder="RC"):
		""" Init the training test sim class and parent classes
		 - omega is the desired output sinusoid frequency. It should correspond to the frequency of the MSN
		 - train_phase is the proportion of time dedicated to training compared to 
		 - output_filename is the filename for storing images of the driving and trained signals
		"""
		super(TrainingSimulation, self).__init__(simulEnv, robot)

		self.omega = omega
		self.transPhase = transPhase
		self.trainPhase = trainPhase
		self.trainingPlot = trainingPlot
		self.wDistPlot = wDistPlot
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

		self.yTraining = self.create_y_training()
		self.xTraining = np.array([])
		self.yTrained = np.array([])

		self.N = self.robot.getState().pos.getArray().shape[1]
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
		y = np.array([])
		for t in self.timeArray:
			line = connections2Array(self.robot.control.modulationFactorTime(t), self.robot.morph.connections)
			if y.size == 0:
				y = line
			else:
				y = np.vstack((y, line))

		return y

	def neuron_fct(self, x):
		""" The transition function of the output neuron """

		return tanh(x)

	def trainStep(self):
		""" Add training data for a given step """

		posArray = self.robot.getState().pos.getArray()
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
		self.yTrained = np.dot(x, w)

		# Print debug
		print " -- Network training by linear regression performed. Sum of residues = {:.4f}".format(res[0]) + \
			". Global NRMSE = {:.4f} --".format(self.nrmse(y, np.dot(x, w)))
		if self.wDistPlot:
			self.plotW()

		# Start Closed-Loop mode
		#self.robot.control.closeLoop()

	def runStep(self):
		""" Run the neuron for a given step """

		# Get position and compute new signal estimation
		posArray = self.robot.getState().pos.getArray()
		y_est = np.asarray(np.dot(posArray, self.weightMatrix))

		# Store estimation in vector
		if self.yTrained.size == 0:
			self.yTrained = y_est
		else:
			self.yTrained = np.vstack((self.yTrained, y_est))

		# Pass the signal estimation to the controller to close the loop
		stepInput = array2Connections(y_est, self.robot.morph.connections)
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
			#print "2. Training phase. It: " + str(it)
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
			if self.trainingPlot:
				self.plot()

	def mse(self, arr1, arr2):
		""" Compute MSE between two matrices """

		assert arr1.shape == arr2.shape, "Mean Square Error can only be computed on matrices with same size"
		a, b = arr2.shape
		return np.sum((arr2 - arr1) ** 2) / float(a * b)

	def nrmse(self, arr1, arr2):
		""" Compute NRMSE between two matrices """

		rmse = np.sqrt(self.mse(arr1, arr2))
		max_val = max(np.max(arr1), np.max(arr2))
		min_val = min(np.min(arr1), np.min(arr2))

		return 1 - (rmse / (max_val - min_val))

	def plot(self, n=None, show=False, save=True):
		""" Print the driving and trained signals as a fct of time in a file"""

		# Compute number of points to display
		if not n or n > self.simulEnv.simulationLength:
			n = self.simulEnv.simulationLength
		n_trans = self.transLength
		n_run = self.runLength
		if n_run > n:
			n_run = n

		# Some arrays init
		nrmse = 0
		y_err = np.zeros(1)

		for i in range(self.O):

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
			plt.plot(self.timeArray[n_trans+1:n-n_run], self.yTrained[1:n-n_run-n_trans:, i] ,"y-", label="Trained sig (train)")
			if n_run != 0:
				plt.plot(self.timeArray[-n_run:], self.yTrained[-n_run:, i] ,"b-", label="Trained sig (run)")
				plt.plot(self.timeArray[-n_run:], y_err[-n_run:] ,"g-", label="Error signal")


			plt.title("Spring control force " + str(i+1) + ". Maximum error =  {:.2f} %".format(100 * np.max(y_err)) + \
				". NRMSE = {:.2f}".format(nrmse))
			Plot.configurePlot(fig, ax, 'Temp', 'Temp', legend=False, legendLocation='lower center')
			plt.xlabel('Simulation step')
			plt.ylabel('Signal value')
			if show: plt.show()
			if save: plt.savefig(self.outputFolder + "/" + self.outputFilename + "_sin_" + str(i+1) + ".png", format='png', dpi=300)
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

class ForceTrainingSimulation(TrainingSimulation):
	""" Extend the TrainingSimulation class to use FORCE online learning """

	def __init__(self, simulEnv, robot, transPhase=0.1, trainPhase=0.5, trainingPlot=True, \
		olPhase=0.5, signTime=None, wDistPlot=True, outputFilename="sinusoid", outputFolder="RC"):
		""" Init class """

		# Fix here the training and running phase if needed
		super(ForceTrainingSimulation, self).__init__(simulEnv, robot, transPhase=transPhase,  \
			trainPhase=trainPhase, trainingPlot=trainingPlot, wDistPlot=wDistPlot, \
			signTime=signTime, outputFilename=outputFilename, outputFolder=outputFolder)

		# Constants
		self.olPhase = olPhase
		self.olLength = int(np.floor(self.olPhase * self.trainLength))
		self.alpha = 0.01

		# Algorithm matrices
		self.trainIt = 0
		self.w = []
		self.p = []
		self.error = np.array([])
		self.yTrained = np.array([])

	def trainStep(self):
		""" Add training data for a given step """

		# Copy input, and output
		x = np.mat(self.robot.getState().pos.getArray().T)
		y = np.mat(self.yTraining[self.iterationNumber])

		# If first iteration, init with random wieghts
		if self.trainIt == 0:
			w = np.random.rand(self.N, self.O)
			p = np.identity(self.N) / self.alpha
			yTrained = np.asarray(w.T * x).T
			self.yTrained = yTrained
			self.error = y - yTrained

		# Else update
		else:

			# Update inverse Correlation Matrix of the network states
			p_prev = np.mat(self.p[-1])
			den = 1 + x.T * p_prev * x
			num = p_prev * x * x.T * p_prev
			p = p_prev - num / den

			# Update weight matrix
			e_prev = np.mat(self.error[-1,:])
			w_prev = np.mat(self.w[-1])
			e_p = w_prev.T * x - y.T
			w = w_prev - p * x * e_p.T
			#print "Error: " + str(np.mean(e_p))

			# Update output
			yTrained =  np.transpose(w.T * x)
			self.yTrained = np.vstack((self.yTrained, np.asarray(yTrained)))

			# Update error
			error = y - yTrained
			self.error = np.vstack((self.error, error))

		# Update iteration
		self.trainIt += 1
		self.w.append(w)
		self.p.append(p)

		# start Closed-Loop mode
		if self.trainIt == self.olLength:
			self.robot.control.closeLoop()

		# Pass the signal estimation to the controller to close the loop
		stepInput = array2Connections(yTrained, self.robot.morph.connections)
		self.robot.control.setStepInput(stepInput)

		return

	def train(self):
		""" Nothing to do here as FORCE is an online method """

		self.weightMatrix = self.w[-1]

		# Start Closed-Loop mode
		print " -- Training phase finished -- "
		#self.robot.control.closeLoop()

		return