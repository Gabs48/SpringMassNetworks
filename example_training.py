"""Train the control signals of a robot to move by optimizing omega, phase, amplitude using GA:"""
from roboTraining.robot import *
from roboTraining.simulate import *
from roboTraining.training import *

env = HardEnvironment()
morph = SpringMorphology(noNodes = 20 ,spring = 100, noNeighbours = 3,environment = env)
control = SineControl(morph)
robot = Robot(morph,control)

plotter = Plotter(plot=False);
simulenv = SimulationEnvironment(1.0 / 200, simulationLength = 100, plot = plotter)

trainscheme = TrainingScheme()
trainscheme.createTrainVariable("omega", 0,10) # omega bounded between 0 and 10
trainscheme.createTrainVariable("phase", 0, 2 * np.pi)
trainscheme.createTrainVariable("amplitude", 0, 0.25)

saver = Save(None, 'RobotData', 'GeneticTraining') 
train = GeneticTraining(trainscheme, robot, simulenv, saver = saver)

param, score = train.run() # perform optimization
train.save() # save all data and plots
