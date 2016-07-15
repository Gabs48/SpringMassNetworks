#! /usr/bin/env python2

import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys,os

from roboTraining.utils import *
from roboTraining.robot import *
from roboTraining.simulate import *
from roboTraining.training import *

class Analysis(object):

	def __init__(self, root="/home/gabs48/edu/Data", folder="test_uniformite"):
		"""Init the analysis class with a folder containing a set of results"""

		self.root = root
		self.folder = folder
		self.path = os.path.join(root, folder)

		self.scores = []
		self.filenames = []

	def load_scores(self):
		"""Browse all folder and retrieve all scores"""

		for path, subdirs, files in os.walk(self.root):
			for name in files:
				if name.find("score") != -1:
					with open(os.path.join(path, name), 'r') as csvfile:
						tab = csv.reader(csvfile, delimiter=';', quotechar='|')
						for row in tab:
							self.scores.append(row)
							self.filenames.append(os.path.join(path, name))

	def plot_scores(self, filename="results_score", show = False, save = True):
		"""Plot score evolution for each file. This can take several minutes"""
		
		i = 0

		for y in self.scores:
			print(" -- Printing score evolution graph for file " + self.filenames[i])
			y = map(float, y)
			x = range((len(y)))
			fig, ax = Plot.initPlot()
			for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
				item.set_fontsize(17)

			barWidth = 0.35
			opacity = 0.4
			plt.bar(x, y, barWidth, alpha = opacity, color = 'k', label = 'distance traveled')
			plt.title("Training Scores     optimum  = " + str(max(y)) )
			Plot.configurePlot(fig, ax, 'Temp', 'Temp', legend = True, legendLocation = 'lower center')
			plt.xlabel('Iteration')
			plt.ylabel('Distance Traveled')
			if show: plt.show()
			if save: plt.savefig(filename + str(i) + ".eps", format = 'eps', dpi = 6000)
			plt.close()
			i += 1

	def get_best_ind(self):
		"""Return best individu score, file path and index"""

		max_t  = 0
		max_path = ""
		max_ind = 0
		i = 0
		print len(self.scores[0])

		for y in self.scores:
			y = map(float, y)
			max_l = max(y)
			if max_l > max_t:
				max_t = max_l
				max_ind = y.index(max_l)
				max_path = self.filenames[i]
			i += 1

		self.max_score = max_t
		self.max_path = max_path
		self.max_ind = max_ind 

		print(' -- Best individu: score: ' + str(max_t) + ' file: ' +  max_path + ' index: ' + str(max_ind))
		return self.max_score, self.max_path, self.max_ind

	def simulate_ind(self, scoreFilename=None, index=None, simName="Simulation"):
		"""Render a simulation movie for a given individu"""

		# Construct robot from config file
		configFilename = scoreFilename.replace("score", "config")
		env = HardEnvironment()
		morph = SpringMorphology(environment=env)
		morph.loadCSV(configFilename)
		control = SineControl(morph)
		control.loadCSV(configFilename)
		robot = Robot(morph,control)

		# Get training parameters of a given individu and update them in the robot
		parameterFilename = scoreFilename.replace("score", "parameter")
		trainscheme = TrainingScheme()
		paramMatrix = trainscheme.loadCSV(parameterFilename, index)
		trainscheme.normalizedMatrix2robot(paramMatrix, robot)

		# Create the simulation
		plotter = Plotter(movie = True, plot = True, movieName = "Simulation", plotCycle = 3)
		simulEnv = SimulationEnvironment(timeStep = 1.0/200, simulationLength = 100, plot =  plotter)
		simul = VerletSimulation(simulEnv, robot)
		print simul.runSimulation();


if __name__ == "__main__":

	# Create Analysis object and load results folder
	an = Analysis()
	an.load_scores()

	# Experiment
	#an.plot_scores()
	score, path, ind = an.get_best_ind()
	an.simulate_ind(path, ind)

