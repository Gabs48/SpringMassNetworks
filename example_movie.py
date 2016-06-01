from roboTraining.robot import *
from roboTraining.simulate import *

"""Create a movie of a moving robot"""

# create robot
env = HardEnvironment() 
morph = SpringMorphology(noNodes = 20,spring = 50, environment = env)
control = SineControl(morph,amplitude = 0.1) 
robot = Robot(morph, control)

# create simulation
plotter = Plotter(movie = True, plot = True, movieName = "RobotMovie", plotCycle = 6, color = True);
simulEnv = SimulationEnvironment(timeStep = 1.0/200, simulationLength = 12000, plot =  plotter)
simul = VerletSimulation(simulEnv, robot)

# let simulation run
simul.runSimulation() # perform simulation
