from robot import SoftEnvironment, HardEnvironment, Morphology, SineControl, RobotState, Robot, SpringMorphology, SpringMorphology3D
from simulate import Plotter, Simulation, SimulationEnvironment, VerletSimulation
from training import Training, TrainingScheme, TrainingVariable, RandomTraining
from utils import SpaceList, Save
import utils
import numpy as np
import unittest

""" test basic functionalities of the RoboTraining Package """

class Constants(object):
	thoroughness = 2; 

def emptyRobot( spring = 0, damping = 0,gravity=0,groundFriction=0,groundContact=0,airFriction=0,amplitude = 0, ground = False):
	""" create a simple robot with two nodes connected by a spring with restlength 1"""
	if ground:
		env = HardEnvironment(gravity = gravity, airFriction = airFriction)
	else:
		env = SoftEnvironment(gravity = gravity, groundFriction = groundFriction, groundContact = groundContact, airFriction = airFriction)
	morph = SpringMorphology(noNodes=2, mass = 1, spring = spring, damping = damping, noNeighbours = 1, environment = env)
	morph.restLength = np.array([[0,1],[1,0]])
	control = SineControl(morph = morph, amplitude = amplitude, phase = np.pi, omega =2* np.pi)
	state = RobotState(0, morph)
	robot = Robot(morph, control, state)
	return robot

def emptyRobot3D( spring = 0, damping = 0,gravity=0,groundFriction=0,groundContact=0,airFriction=0,amplitude = 0, ground = False):
	""" create a simple 3D robot with two nodes connected by a spring with restlength 1"""
	if ground:
		env = HardEnvironment(gravity = gravity, airFriction = airFriction, threeD = true)
	else:
		env = SoftEnvironment(gravity = gravity, groundFriction = groundFriction, groundContact = groundContact, airFriction = airFriction, threeD =True)
	morph = SpringMorphology3D(noNodes=2, mass = 1, spring = spring, damping = damping, noNeighbours = 1, environment = env)
	morph.restLength = np.array([[0,1],[1,0]])
	control = SineControl(amplitude = amplitude, phase = np.pi, omega =2* np.pi, morph = morph)
	state = RobotState(0, morph)
	robot = Robot(morph, control, state)
	return robot

def setState2D (robot, xpos =[0,0], ypos =[0,0] , xspeed =[0,0], yspeed = [0,0]):
	robot.state.setState2D(xpos, ypos, xspeed, yspeed, 0)

def setState3D (robot, xpos =[0,0], ypos =[0,0] , zpos = [0,0] , xspeed =[0,0], yspeed = [0,0], zspeed = [0,0]):
	robot.state.setState3D(xpos, ypos, zpos, xspeed, yspeed, zspeed, 0)

def simpleSimulation(robot, timeStep = 1e-3, simulationLength = 1000, verlet = True):
	""" create a simple simulation without any plotting """
	plotenv = Plotter(plot=False);
	simulenv = SimulationEnvironment(timeStep = timeStep, simulationLength = simulationLength, plot = plotenv, verlet = verlet)
	if verlet:
		simulation = VerletSimulation(simulenv, robot)
	else:
		simulation = Simulation(simulenv, robot)
	return simulation

def plotSimulation(robot, timeStep = 1e-3, simulationLength = 1000, verlet = True, movie = False):
	""" create a simple simulation without any plotting """
	plotenv = Plotter(plot=True, movie = movie);
	simulenv = SimulationEnvironment(timeStep = timeStep, simulationLength = simulationLength, plot = plotenv, verlet = verlet)
	if verlet:
		simulation = VerletSimulation(simulenv, robot)
	else:
		simulation = Simulation(simulenv, robot)
	return simulation

def forceEqual(robot, xforceAim=[0, 0], yforceAim= [0, 0] ):
	""" Are forces equal to the prescribed force"""
	f = robot.computeForces()
	return all((np.allclose(f.x, xforceAim), np.allclose(f.y, yforceAim)))

def forceEqual3D(robot, xforceAim=[0, 0], yforceAim= [0, 0], zforceAim = [0, 0] ):
	""" Are forces equal to the prescribed force"""
	f = robot.computeForces()
	return all(((np.allclose(f.x, xforceAim), np.allclose(f.y, yforceAim)), np.allclose(f.z, zforceAim)))

def stateEqual(robot, xposAim =[0, 0], yposAim = [0, 0], xspeedAim = [0, 0], yspeedAim= [0, 0],tolerance = 1e-3):
	"""Is state is equal to prescribed state"""
	pos,speed, t  = robot.getStateParameters()
	return all (( np.allclose(pos.x, xposAim,atol = tolerance),
		np.allclose(pos.y, yposAim, atol = tolerance),
		np.allclose(speed.x, xspeedAim, atol = tolerance),
		np.allclose(speed.y, yspeedAim, atol = tolerance) ))

def stateEqual3D(robot, xposAim =[0, 0], yposAim = [0, 0], zposAim = [0, 0 ], xspeedAim = [0, 0], yspeedAim= [0, 0], zspeedAim= [0,0], tolerance = 1e-3):
	"""Is state is equal to prescribed state"""
	pos,speed, t  = robot.getStateParameters()
	return all (( np.allclose(pos.x, xposAim,atol = tolerance),
		np.allclose(pos.y, yposAim, atol = tolerance),
		np.allclose(pos.z, zposAim, atol = tolerance),
		np.allclose(speed.x, xspeedAim, atol = tolerance),
		np.allclose(speed.y, yspeedAim, atol = tolerance),
		np.allclose(speed.z, zspeedAim, atol = tolerance)))

class TestRobot(unittest.TestCase):
	def testStaticSpring2D(self):
		"""static spring force 2D"""
		robot = emptyRobot(spring = 10)
		setState2D(robot, xpos= [0,2])
		assert forceEqual(robot, xforceAim = [10, -10])

		setState2D(robot, ypos= [0,3])
		assert forceEqual(robot, yforceAim = [20, -20])

	def testStaticSpring3D(self):
		"""static spring force 3D"""
		robot = emptyRobot3D(spring = 10)

		setState3D(robot, xpos= [0,2])
		assert forceEqual3D(robot, xforceAim = [10, -10])

		setState3D(robot, ypos= [0,3])
		assert forceEqual3D(robot, yforceAim = [20, -20])

		setState3D(robot, zpos= [0,2])
		assert forceEqual3D(robot, zforceAim = [10, -10])

	def testKineticSpring2D(self):
		"""dynamic damping spring force 2D"""
		robot = emptyRobot(damping = 100)

		setState2D(robot, xpos= [0,1], xspeed =[0,2])
		assert forceEqual(robot, xforceAim = [200, -200])

		setState2D(robot, xpos= [0,1], yspeed =[-1, 1])
		assert forceEqual(robot, yforceAim = [200, -200])

	def testKineticSpring3D(self):
		"""dynamic damping spring force 3D"""
		robot = emptyRobot3D(damping = 100)

		setState3D(robot, xpos= [0,1], xspeed =[0,2])
		assert forceEqual3D(robot, xforceAim = [200, -200])

		setState3D(robot, xpos= [0,1], yspeed =[-1, 1])
		assert forceEqual3D(robot, yforceAim = [200, -200])

		setState3D(robot, xpos= [0,1], zspeed =[0,2])
		assert forceEqual3D(robot, zforceAim = [200, -200])

	def testGravity(self):
		""" Gravity Force"""
		robot = emptyRobot(gravity = 10)
		assert forceEqual(robot, yforceAim = [-10, -10])

	def testAirfriction(self):
		""" Air Friction Force"""
		robot = emptyRobot(airFriction = 10)
		setState2D(robot, xpos= [0, 1], xspeed =[-5, 5], yspeed= [-1, 1])
		assert forceEqual(robot, xforceAim = [50, -50], yforceAim = [10, -10])

	def testAirfriction3D(self):
		""" Air Friction Force"""
		robot = emptyRobot3D(airFriction = 10)
		setState3D(robot, xpos= [0, 1], xspeed =[-5, 5], yspeed= [-1, 1], zspeed = [10, 10])
		assert forceEqual3D(robot, xforceAim = [50, -50], yforceAim = [10, -10], zforceAim = [-100, -100])	
		
	def testNormalforce(self):
		""" Normal Force """
		robot = emptyRobot(groundContact = 1)
		robot.state.setState2D([0, 1], [-1, -1] , [0, 0], [-1, 1], 0)
		f = robot.computeForces()
		assert np.allclose(f.x, [0, -0])
		assert np.alltrue(f.y > [0, 0])

	def testFriction(self):
		""" Friction Force """
		robot = emptyRobot(groundFriction = 1)
		setState2D(robot, [0, 1], [-1, 1] , [4, 4], [-1, 1])
		f = robot.computeForces()
		assert f.x[0] < 0
		assert f.x[1] == 0
		assert np.allclose(f.y , [0, 0])

	def testdefault(self):
		""" default robot calculates force """
		morph = SpringMorphology()
		control = SineControl(morph)
		robot = Robot(morph, control);
		f = robot.computeForces()
		assert all(np.isfinite(f.x))
		assert all(np.isfinite(f.y))

	def testCopyState(self):
		"""create a deep copy of the state"""
		robot = emptyRobot();
		setState2D(robot, xpos = [0, 1])
		state = robot.getState()
		robot.state.pos.x[1] = 4
		assert state.pos.x[1] == 1

	def testCopyState3D(self):
		"""create a deep copy of the state"""
		robot = emptyRobot3D();
		setState3D(robot, zpos = [0, 1])
		state = robot.getState()
		robot.state.pos.z[1] = 4
		assert state.pos.z[1] == 1

	def testControl(self):
		robot = emptyRobot3D( amplitude = 0.5)
		assert np.allclose(robot.control.modulationFactor(robot.state), [[1, 1], [1, 1]])

		robot.state.currentTime = 0.25
		assert np.allclose(robot.control.modulationFactor(robot.state), [[1, 0.5], [0.5, 1]])

		robot.state.currentTime = 0.75
		assert np.allclose(robot.control.modulationFactor(robot.state), [[1, 1.5],[1.5, 1]])

class TestSimulation(unittest.TestCase):
	
	def testSinusX(self):
		"""Robot behaves as harmonic oscillator with period 2*Pi """
		if Constants.thoroughness >= 2:
			robot = emptyRobot(spring = 0.5)
			timestep = 1.0 / 1000; 
			halfPeriod = int (1* np.pi * 1000)
			simulation = simpleSimulation(robot,timestep,halfPeriod)
			tolerance = 1e-3
			"""
			# X direction
			setState2D(robot, xpos = [0, 1.5])
			# half period
			simulation.runSimulation()
			assert stateEqual(robot, xposAim = [0.5, 1])
			# full period
			simulation.runSimulation();
			assert stateEqual(robot, xposAim = [0, 1.5])

			# Y direction
			setState2D(robot, ypos = [0, 1.5])
			# half period
			simulation.runSimulation();
			assert stateEqual(robot, yposAim = [0.5, 1])
			# full period
			simulation.runSimulation();
			assert stateEqual(robot, yposAim = [0, 1.5])
			"""
			# Z direction
			robot = emptyRobot3D(spring = 0.5)
			simulation.robot = robot
			setState3D(robot, zpos = [0, 1.5])
			# half period
			simulation.runSimulation();
			assert stateEqual3D(robot, zposAim = [0.5, 1])
			# full period
			simulation.runSimulation();
			assert stateEqual3D(robot, zposAim = [0, 1.5])

		else: print "testthoroughness is set too low for this test"

class Testutils(unittest.TestCase):
	def testArray2Connections(self):
		"""conversion from an array to the connections matrix and back"""
		robot = emptyRobot()
		array = [10]
		connections = utils.array2Connections(array, robot.getConnections())
		arrayAccent = utils.connections2Array(connections, robot.getConnections())
		assert np.allclose(array, arrayAccent)

class TestTraining(unittest.TestCase):
	def testNormalize(self):
		"""normalization  and denormalization procedure of TrainVariable """
		trainvar = TrainingVariable("spring",0,1000)
		testArray = [500, 300, 3.2 , 0]
		testArraynorm = trainvar.normalize(testArray)
		testArrayAccent = trainvar.denormalize(testArraynorm)
		assert np.allclose(testArray, testArrayAccent)

	def testSetterSpring(self):
		"""array to robot and back"""
		trainScheme = TrainingScheme();
		trainScheme.createTrainVariable("spring", 0, 1000)
		robot = emptyRobot();
		
		# normal test
		array = np.array([[0.4]])
		trainScheme.normalizedMatrix2robot(array, robot)
		arrayAccent = trainScheme.robot2normalizedMatrix(robot)
		assert np.allclose(array, arrayAccent)

		# check whether exceptions are thrown in case of invalid input
		with self.assertRaises(AssertionError):
			array = np.array([[0.4, 0.4]])
			trainScheme.normalizedMatrix2robot(array, robot)

		with self.assertRaises(AssertionError):
			array = np.array([0.4])
			trainScheme.normalizedMatrix2robot(array, robot)

		with self.assertRaises(AssertionError):
			array = np.array([5])
			trainScheme.normalizedMatrix2robot(array, robot)

	def testCreateTraining(self):
		""" no exceptions may be thrown """
		if Constants.thoroughness >= 2:
			env=SoftEnvironment()
			morph=SpringMorphology(noNodes = 10,spring = 1000, noNeighbours = 3,environment = env)
			control=SineControl(morph)
			state=RobotState(0,morph)
			robot=Robot(morph,control,state)

			plotter =Plotter(plotCycle=50,plot=False);
			simulenv=SimulationEnvironment(timeStep = 0.0005,simulationLength=2000,plot =plotter)

			simul = Simulation(simulenv,robot)
			simul.runSimulation()

			trainscheme = TrainingScheme()
			trainscheme.createTrainVariable("spring",0,3000)
			trainscheme.createTrainVariable("phase",0,2*np.pi)

			training=RandomTraining(trainscheme,robot,simulenv)
			trainSave = Save(training, 'temp', 'default')
			trainSave.save([[10,10],[20,20]])
		else: print "testthoroughness is set too low for this test"

class TestSpaceList(unittest.TestCase):
	space2Da = SpaceList(np.array([[1],[2]]))
	space2Db = SpaceList(np.array([4]),np.array([10]))
	space2Dc = SpaceList(np.array([[1,2],[6,15]]))
	space3Da = SpaceList(np.array([[1],[2.0],[3]]))
	space3Db = SpaceList(np.array([10.0]),np.array([100]),np.array([1000.0]))
	space3Dc = SpaceList(np.array([[1, 2 , 3, 4],[10 , 20 , 30, 40],[100, 200, 300, 400]]))
	array = np.array([1,2,3,4])

	def testAdd2D(self):
		sum = self.space2Da + self.space2Db
		assert sum.x == 5
		assert sum.y == 12

	def testAdd3D(self):
		sum = self.space3Da + self.space3Db
		assert sum.x == 11
		assert sum.y == 102
		assert sum.z == 1003

	def testAddCopy3D(self):
		copy = self.space3Da.copy()
		copy += self.space3Da
		assert copy.x == 2
		assert copy.y == 4
		assert copy.z == 6
		assert self.space3Da.x == 1
		assert self.space3Da.y == 2
		assert self.space3Da.z == 3

	def testMult3D(self):
		product = self.space3Da * self.space3Db
		assert product.x == 10
		assert product.y == 200
		assert product.z == 3000

	def testMult2DCopy(self):
		copy = self.space2Da.copy()
		copy *= self.space2Da
		assert copy.x == 1
		assert copy.y == 4
		assert self.space2Da.x == 1
		assert self.space2Da.y == 2

	def testMult3Darray(self):
		product= self.array * self.space3Dc 
		aim = np.array([1, 4, 9, 16])
		assert np.all(product.x == aim)
		assert np.all(product.y == aim * 10)
		assert np.all(product.z == aim * 100)

	def testMult3Dscalar(self):
		product= 4 * self.space3Dc 
		aim = 4 * np.array([1, 2, 3, 4])
		assert np.all(product.x == aim)
		assert np.all(product.y == aim * 10)
		assert np.all(product.z == aim * 100)

	def testdiff2D(self):
		xdiff, ydiff = self.space2Dc.getDifference()
		assert np.all( xdiff == [[0, -1],[1, 0]])
		assert np.all( ydiff == [[0,- 9],[9, 0]])

def run(verbosity = 2, thorogouhness = 1):
	Constants.thoroughness = thorogouhness
	suite = unittest.TestSuite();
	suite.addTests(unittest.makeSuite(TestRobot))
	suite.addTests(unittest.makeSuite(TestSimulation))
	suite.addTests(unittest.makeSuite(TestTraining))
	suite.addTests(unittest.makeSuite(Testutils))
	suite.addTests(unittest.makeSuite(TestSpaceList))
	unittest.TextTestRunner(verbosity = verbosity).run(suite)

def runSpecial(verbosity = 2):
	suite = unittest.TestSuite();
	suite.addTest(TestRobot("testControl"))
	unittest.TextTestRunner(verbosity = verbosity).run(suite)

if __name__ == '__main__':
	unittest.main()