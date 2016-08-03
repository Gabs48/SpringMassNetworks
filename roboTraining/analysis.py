from utils import *
from robot import *
from simulate import *
from training import *

# import jsonpickle
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
from matplotlib.mlab import *
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter
import sys,os

class Analysis(object):

	def __init__(self, root="/home/gabs48/edu/Data", folder="test_uniformite"):
		"""Init the analysis class with a folder containing a set of results"""

		self.root = root
		self.folder = folder
		self.path = os.path.join(root, folder)

		self.scores = []
		self.filenames = []
		self.parameters = []
		self.maxomega = []
		self.maxampli = []
		self.opt_type = []
		self.sim_time = []
		self.k = []
		self.m = []
		self.n_nodes = []
		self.ps = []

		self.y = []
		self.x = []
		self.x_av = []
		self.x_std = []
		self.x_sce_f = []
		self.y_av = []
		self.y_std_a = []
		self.y_std_b = []
		self.y_std = []
		self.y_min = []
		self.y_max = []
		self.y_sce = []
		self.y_sce_f = []
		self.pc1 = []
		self.pc2 = []
		self.pc1_av = []
		self.pc2_av = []
		self.pc1_std = []
		self.pc2_std = []
		self.window = []

	def _load_parameters(self):
		"""Load the training parameters from all parameters files"""

		for i, f in enumerate(self.filenames):
			with open(f.replace("score", "parameter"), 'r') as csvfile:
				tab = list(csv.reader(csvfile, delimiter=';', quotechar='|'))
			
				params = []
				for j in range(len(tab)) :
					if j%5 == 2:
						row = tab[j]
					if j%5 == 3 or j%5 == 4:
						row.extend(tab[j])
					if j%5 == 0 and j != 0:
						row_float = list(map(float, row))
						params.append(row_float)

				row_float = list(map(float, row))
				params.append(row_float) # last iteration
				self.parameters.append(params)
				print(" -- Parameters of " + str(i+1) + "/" + str(len(self.filenames)) + " loaded -- ")

	def _load_configs(self):
		"""Load configuration parameters from all the config files"""

		for f in self.filenames:
			with open(f.replace("score", "config"), 'r') as csvfile:
				tab = list(csv.reader(csvfile, delimiter=';', quotechar='|'))

				# Find max omega and amplitudes
				params_ind = findIndex(tab, "trainableParams:")
				self.maxomega.append(float(tab[params_ind[0] + 3][params_ind[1] + 3]))
				self.maxampli.append(float(tab[params_ind[0] + 9][params_ind[1] + 3]))

				# Find the optimization type
				rand_type_ind = findIndex(tab, "noInstances:")
				cma_type_ind = findIndex(tab, "maxIter:")
				if rand_type_ind != [-1, -1]:
					self.opt_type.append("RANDOM")
				elif cma_type_ind != [-1, -1]:
					self.opt_type.append("CMA")

				# Find simulation time
				ts_ind = findIndex(tab, "timeStep:")
				sl_ind = findIndex(tab, "simulationLength:")
				self.sim_time.append(float(tab[ts_ind[0]][ts_ind[1] + 1])*float(tab[sl_ind[0]][sl_ind[1] + 1]))

				# Find population size
				ps_ind = findIndex(tab, "popSize:")
				if ps_ind == [-1, -1]:
					self.ps.append(19)
				else:
					self.ps.append(float(tab[ps_ind[0]][ps_ind[1] + 1]))

				# Find Nodes number
				nn_ind = findIndex(tab, "noNodes:")
				self.n_nodes.append(int(tab[nn_ind[0]][nn_ind[1] + 1]))

				# Find mass
				m_ind = findIndex(tab, "mass:")
				self.m.append(float(tab[m_ind[0]][m_ind[1] + 1]))

				# Find spring constant
				k_ind = findIndex(tab, "spring:")
				self.k.append(float(tab[k_ind[0]+1][k_ind[1] + 2]))	 

	def _compute_stats(self, window=None, pca=False):
		"""Compute statistics of list of scores"""

		window_init = window

		if pca:
			# Loading parameters
			if not self.parameters:
				print(" -- Loading parameter files -- ")
				self._load_parameters()
			print(" -- Parameter files loaded -- ")

		if not self.y_max or pca:
			i = 0
			for y in self.y:

				# Compute average
				if window_init == None:
					window = len(self.x[i]) / 40
				y_av = np.convolve(np.array(y), np.ones((window,))/window, mode='valid')
				if window%2 == 0:
					x_av = self.x[i][window/2:len(self.x[i])-window/2+1]
				else:
					x_av = self.x[i][window/2:len(self.x[i])-window/2]

				# Compute std deviation
				y_std = self._window_stdev(np.array(y), window=window)
				x_std = self.x[i][window:len(self.x[i])-window+1]
				if window%2 == 0:
					y_std_a = y_av[window/2:len(y_av)-window/2] + y_std
					y_std_b = y_av[window/2:len(y_av)-window/2] - y_std
				else:
					y_std_a = y_av[window/2:len(y_av)-window/2-1] + y_std
					y_std_b = y_av[window/2:len(y_av)-window/2-1] - y_std

				# Compute min and max
				val_max = y[0]
				val_min = y[0]
				y_max = []
				y_min = []
				for val in y:
					if val_max < val:
						val_max = val
					if val_min > val:
						val_min = val
					y_max.append(val_max)
					y_min.append(val_min)

				# Compute square convergence error
				if window%2 == 0:
					y_max_red = np.array(y_max[window/2:len(y_max)-window/2+1])
				else:
					y_max_red = np.array(y_max[window/2:len(y_max)-window/2])
				y_sce = np.sqrt(((y_av - y_max_red) ** 2))/y_max_red
				window2 = 2 * window
				if window2%2 == 0:
					x_sce_f = np.array(self.x[i][window2/2:len(y_sce)-window2/2+1])
				else:
					x_sce_f = np.array(self.x[i][window2/2:len(y_sce)-window2/2])
				y_sce_f = np.convolve(np.array(y_sce), np.ones((window2,))/window2, mode='valid')

				# Perform parameters PCA
				if pca:
					vec = np.array(self.parameters[i])
					if vec.shape[0] > vec.shape[1]:
						res = PCA(vec)
						pc1 = res.Y[:,0]
						pc2 = res.Y[:,1]
					else:
						pc1 = vec[:,0]
						pc2 = vec[:,1]

					pc1_av = np.convolve(pc1, np.ones((window,))/window, mode='valid')
					pc2_av = np.convolve(pc2, np.ones((window,))/window, mode='valid')
					if window%2 == 0:
						pc1_std = pc1_av[window/2:len(pc1_av)-window/2]
						pc2_std = pc2_av[window/2:len(pc2_av)-window/2]
					else:
						pc1_std = pc1_av[window/2:len(pc1_av)-window/2-1]
						pc2_std = pc2_av[window/2:len(pc2_av)-window/2-1]

					self.pc1.append(pc1)
					self.pc2.append(pc2)
					self.pc1_av.append(pc1_av)
					self.pc2_av.append(pc2_av)
					self.pc1_std.append(pc1_std)
					self.pc2_std.append(pc2_std)

				self.window.append(window)
				self.y_av.append(y_av)
				self.x_std.append(x_std)
				self.y_std_a.append(y_std_a)
				self.y_std_b.append(y_std_b)
				self.y_std.append(y_std)
				self.x_av.append(x_av)
				self.y_max.append(y_max)
				self.y_min.append(y_min)
				self.y_sce.append(y_sce)
				self.y_sce_f.append(y_sce_f)
				self.x_sce_f.append(x_sce_f)

				i += 1

	def _rearrange_pop(self, index=0):
		"""Group a score list in generations and return gen number min, max and average"""

		dim = len(self.y[index])
		ps = self.ps[index]
		assert dim%ps == 0, "The total number of iteration (" + str(dim) + ") shall be a " + \
			"multiple of the population size (" + str(ps) + "). Please verify the file " +  \
			self.filenames[index] + " or this sript!"

		array = np.array(self.y[index])
		matrix = np.reshape(array, (-1, ps))
		y_min = np.min(matrix, axis=1)
		y_max = np.max(matrix, axis=1)
		y_av = np.mean(matrix, axis=1)
		x = np.array(range((y_av.size)))

		return x, y_min, y_max, y_av

	def _window_stdev(self, arr, window=50):
		"""Compute std deviation of an array with a sliding window"""

		c1 = uniform_filter(arr, window*2, mode='constant', origin=-window)
		c2 = uniform_filter(arr*arr, window*2, mode='constant', origin=-window)
		res = ((c2 - c1*c1)**.5)[:-window*2+1]

		return res

	def load(self):
		"""Browse all folder and retrieve all scores. Then load configs and parameters"""

		# Load scores
		for path, subdirs, files in os.walk(self.path):
			for name in files:
				if name.find("score") != -1 and os.path.splitext(name)[1] == ".csv":
					with open(os.path.join(path, name), 'r') as csvfile:
						tab = csv.reader(csvfile, delimiter=';', quotechar='|')
						for row in tab:
							self.scores.append(row)
							self.filenames.append(os.path.join(path, name))
		
		# Fill score list and iteration range
		for y in self.scores:
			self.y.append(list(map(float, y)))
			self.x.append(range((len(y))))

		# Load configs and parameters
		self._load_configs()

	def plot_score(self, index=0, filename="results_score", title=None, show=False, save=True):
		"""Plot score evolution for a given file"""
		
		print(" -- Printing score bar graph for file " + self.filenames[index])

		fig, ax = Plot.initPlot()
		for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
			item.set_fontsize(17)

		barWidth = 0.35
		opacity = 0.4
		plt.bar(self.x[index], self.y[index], barWidth, alpha = opacity, color = 'k', label = 'distance traveled')
		if title != None:
			plt.title(title)
		else:
			plt.title("Training Scores (optimum  = " + str(max(self.y[index])) + ")")
		Plot.configurePlot(fig, ax, 'Temp', 'Temp', legend = True, legendLocation = 'lower center')
		plt.xlabel('Iteration')
		plt.ylabel('Distance Traveled')
		if show: plt.show()
		if save: plt.savefig(filename + ".png", format='png', dpi=300)
		plt.close()

	def plot_gen(self, index=0, filename="results_gen", title=None, show=False, save=True):
		"""Plot scores rearranged in generations for a given file"""

		if self.opt_type[index] != "RANDOM":
			x, y_min, y_max, y_av = self._rearrange_pop(index)

			print(" -- Printing generation scores for file " + self.filenames[index])

			fig, ax = Plot.initPlot()
			for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
				item.set_fontsize(17)

			plt.plot(x, y_max ,"r-", label="Gen max")
			plt.plot(x, y_av, "g-", label="Gen av")
			plt.plot(x, y_min, "b-", label="Gen min")
			if title != None:
				plt.title(title)
			else:
				plt.title("Training scores of " + self.opt_type[index] + " algorithm with popSize = " + \
				num2str(self.ps[index]))
			Plot.configurePlot(fig, ax, 'Temp', 'Temp', legend=True, legendLocation='lower center')
			plt.xlabel('Generation number')
			plt.ylabel('Distance Traveled')
			if show: plt.show()
			if save: plt.savefig(filename + ".png", format='png', dpi=300)
			plt.close()

		else:
			print(" -- Can't print generation scores for file " + self.filenames[index] + \
				" with "+ self.opt_type[index] + " optimization type.")

	def plot_score_av(self, index=0, filename="results_score_av", title=None, show=False, save=True):
		"""Plot average and max score evolution for a given file"""

		if not self.y_max:
			self._compute_stats()

		print(" -- Printing score stats graph for file " + self.filenames[index])

		fig, ax = Plot.initPlot()
		for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
			item.set_fontsize(17)

		plt.plot(self.x[index], self.y_min[index] ,"r-", label="maximum")
		plt.plot(self.x_std[index], self.y_std_a[index] ,"r--", linewidth=0.2, label="max std dev")
		plt.plot(self.x_av[index], self.y_av[index], "g-", label="average score")
		plt.plot(self.x_std[index], self.y_std_b[index] ,"b--", linewidth=0.2, label="min std dev")
		plt.plot(self.x[index], self.y_max[index], "b-", label="minimum")
		if title != None:
			plt.title(title)
		else:
			plt.title("Training max and average (optimum  = " + str(max(self.y[index])) + ")")
		Plot.configurePlot(fig, ax, 'Temp', 'Temp', legend=True, legendLocation='lower center')
		plt.xlabel('Iteration')
		plt.ylabel('Distance Traveled')
		if show: plt.show()
		if save: plt.savefig(filename + ".png", format='png', dpi=300)
		plt.close()

	def plot_conv_err(self, index=0, filename="results_score_conv_err", title=None, window=None, show=False, save=True):
		"""Plot convergence error for a given file"""

		if not self.y_max:
			self._compute_stats()

		print(" -- Printing convergence error graph for file " + self.filenames[index])

		fig, ax = Plot.initPlot()
		for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
			item.set_fontsize(17)

		plt.plot(self.x_av[index], self.y_sce[index] ,"b-", linewidth=0.2, label="Convergence error")
		plt.plot(self.x_sce_f[index], self.y_sce_f[index] ,"r-", linewidth=0.8, label="Averaged convergence error")
		Plot.configurePlot(fig, ax, 'Temp', 'Temp', legend=True, legendLocation='upper center')
		if title != None:
			plt.title(title)
		else:
			plt.title("Training convergence error (optimum  = " + str(max(self.y[index])) + ")")
		plt.xlabel('Iteration')
		plt.ylabel('Convergence square error')
		if show: plt.show()
		if save: plt.savefig(filename + ".png", format='png', dpi=300)
		plt.close()

	def plot_state_space(self, index=0, filename="results_state_space", title=None, show=False, save=True):
		"""Plot the score evolution in a state space composed by the two first PC to 
		get a glance at the dynamics of the problem"""

		if not self.parameters:
			self._compute_stats(pca=True)

		# Get windows and gap sizes
		n = self.pc1[index].shape[0]
		n_av = self.pc1_av[index].shape[0]
		if self.window[index] > n:
			window = n
		else:
			window = self.window[index]
		gap = n / window
		gap_av = n_av / window

		# Plot full trajectories
		fig, ax = Plot.initPlot(proj="3d")
		for j in xrange(1, window):
			ax.plot(self.pc1[index][j*gap:(j+1)*gap], self.pc2[index][j*gap:(j+1)*gap], self.y[index][j*gap:(j+1)*gap], \
				c=plt.cm.jet(1.*j/window), linewidth=0.2, label="PCA trajectory")
		if title != None:
			plt.title(title)
		else:
			plt.title("Score in fct of the 2 first PC parameters with " + self.opt_type[index] + " training and " + \
				num2str(self.sim_time[index]) + " s simulations")
		ax.set_xlabel('PC 1')
		ax.set_ylabel('PC 2')
		ax.set_zlabel('Score')
		print(" -- Trajectory image "+ str(index+1) + "/" + str(len(self.y)) + " generated -- ")
		if show: plt.show(block=True)
		if save: plt.savefig(filename + ".png", format='png', dpi=300)
		plt.close()

		# Plot averaged trajectories
		fig2, ax2 = Plot.initPlot(proj="3d")
		for j in xrange(1, window):
			ax2.plot(self.pc1_av[index][j*gap_av:(j+1)*gap_av], self.pc2_av[index][j*gap_av:(j+1)*gap_av], \
				self.y_av[index][j*gap_av:(j+1)*gap_av], ".", c=plt.cm.jet(1.*j/window), \
				linewidth=1, markersize=0.4, label="PCA average")
		ax2.plot(self.pc1_std[index], self.pc2_std[index], self.y_std_a[index],	"r-", linewidth=0.4, label="PCA max std")
		ax2.plot(self.pc1_std[index], self.pc2_std[index], self.y_std_b[index],	"b-", linewidth=0.4, label="PCA min std")
		if title != None:
			plt.title("Averaged " + title)
		else:
			plt.title("Averaged score in fct of the 2 first PC parameters with " + self.opt_type[index] + " training and " + \
				num2str(self.sim_time[index]) + " s simulations")
		ax.set_xlabel('PC 1')
		ax.set_ylabel('PC 2')
		ax.set_zlabel('Average score')
		print(" -- Averaged trajectory image " + str(index+1) + "/" + str(len(self.y)) + " generated -- ")
		if show: plt.show(block=True)
		if save: plt.savefig(filename + "_av.png", format='png', dpi=300)
		plt.close()

	def plot_all_scores(self, filename="results_score", show=False, save=True):
		"""Plot score evolution for a all files. This can take several minutes"""

		i = 0
		for y in self.y:
			self.plot_score(index=i, filename=filename + "_" + str(i), show=show, save=save)
			i += 1

	def plot_all_gens(self, filename="results_gen", show=False, save=True):
		"""Plot score evolution rearranged in generations for all files."""

		i = 0
		for y in self.y:
			self.plot_gen(index=i, filename=filename + "_" + str(i), show=show, save=save)
			i += 1

	def plot_all_scores_av(self, filename="results_score_av", show=False, save=True):
		"""Plot average and max score evolution for a all files."""

		i = 0
		for y in self.y:
			self.plot_score_av(index=i, filename=filename + "_" + str(i), show=show, save=save)
			i += 1

	def plot_all_conv_errs(self, filename="results_score_conv_err", show=False, save=True):
		"""Plot convergence error for all files"""

		i = 0
		for y in self.y:
			self.plot_conv_err(index=i, filename=filename + "_" + str(i), show=show, save=save)
			i += 1

	def plot_all_state_spaces(self, filename="results_state_space", show=False, save=True):
		"""Plot the score evolution in a state space composed by the two first PC to 
		get a glance at the dynamics of the problem"""

		print(" -- Printing score evolution in principal components parameter space. This can take a while -- ")

		i = 0
		for y in self.y:
			self.plot_state_space(index=i, filename=filename + "_" + str(i), show=show, save=save)
			i += 1

	def get_best_ind(self, index=None):
		"""Return best individu score, file index and place index"""

		max_t  = 0
		max_index1 = 0
		max_index2 = 0
		i = 0

		if index == None:
			for y in self.y:
				max_l = max(y)
				if max_l > max_t:
					max_t = max_l
					max_index2 = y.index(max_l)
					max_index1 = i
				i += 1

			self.max_score = max_t
			self.max_index1 = max_index1
			self.max_index2 = max_index2

			print(' -- Best individu: score: ' + str(max_t) + ' file: ' +  self.filenames[max_index1] + \
				' (' + str(max_index2) + "/" + str(len(self.scores[max_index1])) + ") --")

			return self.max_score, self.max_index1, self.max_index2

		else:
			max_index1 = index
			max_t = max(self.y[index])
			max_index2 = self.y[index].index(max_t)

			return max_t, max_index1, max_index2

	def simulate_ind(self, time=2, index1=None, index2=None, simName="Simulation"):
		"""Render a simulation movie for a given individu"""

		scoreFilename = self.filenames[index1]
		print(' -- Simulate individu ' +  scoreFilename + ' (' + str(index2) + "/" + \
			str(len(self.scores[index1])) + ") with score " + num2str(self.y[index1][index2]) + \
			" for " + num2str(time) + "s. This can takes several minutes. --")

		# Construct robot from config file
		configFilename = scoreFilename.replace("score", "config")
		env = HardEnvironment()
		morph = SpringMorphology(noNodes=20 , spring=100, noNeighbours=3, environment=env)
		# morph.loadCSV(configFilename)
		control = SineControl(morph)
		control.loadCSV(configFilename)
		robot = Robot(morph, control)

		# Get training parameters of a given individu and update them in the robot
		parameterFilename = scoreFilename.replace("score", "parameter")
		trainscheme = TrainingScheme()
		paramMatrix = trainscheme.loadCSV(parameterFilename, index2)
		trainscheme.normalizedMatrix2robot(paramMatrix, robot)

		# Create the simulation
		plotter = Plotter(movie=True, plot=True, movieName=simName, plotCycle = 6)
		simulEnv = SimulationEnvironment(timeStep=1.0/200, simulationLength=200*time, plot=plotter, \
			perfMetr="dist", controlPlot=False)

		# jsonpickle.set_encoder_options('simplejson', sort_keys=True, indent=4)
		# with open("simulEnv.json", 'wb') as f:
		# 	f.write(jsonpickle.encode(simulEnv))
		# 	f.close()
		# with open("robot.json", 'wb') as f:
		# 	f.write(jsonpickle.encode(robot))
		# 	f.close()

		# Do the simulation
		simul = Simulation(simulEnv, robot)
		score = simul.runSimulation();

		print(" -- Simulation terminated with score " + num2str(score) + \
			". Video saved in file " + simName + ".mp4 --")

	## ------------------- Specific Simulation types ---------------------------

	def simtime(self):
		"""Perform specific analysis for a simtime batch"""

		folder = "simtime_pic/"
		mkdir_p(folder)

		i = 0
		for score in self.scores:
			
			ext = self.opt_type[i] + "_" + num2str(self.sim_time[i]) + "s"
			err_title = "Convergence error evolution with " + self.opt_type[i] + " training and " + \
				num2str(self.sim_time[i]) + " s simulations"
			evo_title = "Averaged optimization evolution with " + self.opt_type[i] + " training and " + \
				num2str(self.sim_time[i]) + " s simulations"
			gen_title = "Generation evolution with " + self.opt_type[i] + " training and " + \
				num2str(self.sim_time[i]) + " s simulations"
			ss_title = "PC parameters evolution with " + self.opt_type[i] + " training and " + \
				num2str(self.sim_time[i]) + " s simulations"
			err_filename = folder + "err_" + ext
			evo_filename = folder + "evol_" + ext
			gen_filename = folder + "gen_" + ext
			ss_filename = folder + "exp_" + ext
			self.plot_score_av(index=i, filename=evo_filename, title=evo_title)
			self.plot_conv_err(index=i, filename=err_filename, title=err_title)
			self.plot_gen(index=i, filename=gen_filename, title=gen_title)
			self.plot_state_space(index=i, filename=ss_filename, title=ss_title)
			i += 1

	def km(self):
		"""Perform specific analysis for a km batch"""

		folder = "km_pic/"
		mkdir_p(folder)

		for i, score in enumerate(self.scores):
			ext = num2str(self.k[i]) + "_" + num2str(self.m[i])
			err_title = "Convergence error evolution with k=" + num2str(self.k[i]) + " and m=" + \
				num2str(self.m[i])
			evo_title = "Averaged optimization evolution with k=" + num2str(self.k[i]) + " and m=" + \
				num2str(self.m[i])
			gen_title = "Generation evolution with k=" + num2str(self.k[i]) + " training and m=" + \
				num2str(self.m[i])
			ss_title = "PC parameters evolution with k=" + num2str(self.k[i]) + " training and m=" + \
				num2str(self.m[i])
			err_filename = folder + "err_" + ext
			evo_filename = folder + "evol_" + ext
			gen_filename = folder + "gen_" + ext
			ss_filename = folder + "exp_" + ext
			self.plot_score_av(index=i, filename=evo_filename, title=evo_title)
			self.plot_conv_err(index=i, filename=err_filename, title=err_title)
			self.plot_gen(index=i, filename=gen_filename, title=gen_title)
			self.plot_state_space(index=i, filename=ss_filename, title=ss_title)

	def nodes(self):
		"""Perform specific analysis for a nodes batch"""

		folder = "nodes_pic/"
		mkdir_p(folder)

		for i, score in enumerate(self.scores):
			ext = num2str(self.n_nodes[i])
			err_title = "Convergence error evolution with nodes number =" + ext
			evo_title = "Averaged optimization evolution with nodes number =" + ext
			gen_title = "Generation evolution with nodes number =" + ext
			ss_title = "PC parameters evolution with nodes number =" + ext
			err_filename = folder + "err_" + ext
			evo_filename = folder + "evol_" + ext
			gen_filename = folder + "gen_" + ext
			ss_filename = folder + "exp_" + ext
			self.plot_score_av(index=i, filename=evo_filename, title=evo_title)
			self.plot_conv_err(index=i, filename=err_filename, title=err_title)
			self.plot_gen(index=i, filename=gen_filename, title=gen_title)
			self.plot_state_space(index=i, filename=ss_filename, title=ss_title)

	def pareto(self, filename="results_pareto", show=False, save=True):
		"""Perform specific analysis for a pareto batch"""
		
		print(" -- Pareto Analysis of folder " + self.path + "--")

		dist = []
		omega = []
		ampli = []
		omega_res = []
		norm_dist = []
		norm_power = []
		k_spring = 100
		mass = 1
		sim_time = self.sim_time[0]
		opt_type = self.opt_type[0]

		# Fill values from loaded variables
		for i in range(len(self.y)):
			duplicate = False
			assert self.sim_time[i] == sim_time, \
				"For a meaningfull pareto graph, ensure the simulation times are the same for all data"
			assert self.opt_type[i] == opt_type, \
				"For a meaningfull pareto graph, ensure the optimization algorithms are the same for all data"

			# If couple omega/ampli already existsn average with previous one
			for j, om in enumerate(omega):
				for k, am in enumerate(ampli):
					if om == self.maxomega[i] and am == self.maxampli[i] and j == k:
						dist[j].append(self.get_best_ind(index=i)[0])
						duplicate = True

			if duplicate == False:
				dist.append([self.get_best_ind(index=i)[0]])
				omega.append(self.maxomega[i])
				ampli.append(self.maxampli[i])
				omega_res.append(np.sqrt(k_spring / mass))

		# Average points with multiple values
		n_av = []
		for i in range(len(dist)):
			if  len(dist[i]) > 1:
				n_av.append(len(dist[i]))
			dist[i] = sum(dist[i]) / len(dist[i])
		if len(n_av) != 0:
			print(" -- Averaging " + str(len(n_av)) + " graph points with on average " + \
				num2str(float(sum(n_av)/len(n_av))) + " data sets for each --")

		# Sort lists
		ampli, omega, dist = (list(t) for t in zip(*sorted(zip(ampli, omega, dist))))

		# Compute reference power and distance
		max_omega = max(omega)
		max_ampli = max(ampli)
		n_ampli = len(ampli) / len(set(ampli))
		n_omega = len(omega) / len(set(omega))
		power_0 = max_omega * max_ampli ** 2
		dist_0 = sum(dist[-n_ampli:len(ampli)])/ n_ampli

		# Compute each point power and distance
		for i in range(len(ampli)):
			norm_power.append(ampli[i] ** 2 * omega[i] / power_0)
			norm_dist.append(dist[i] / dist_0)
		omega_sorted, norm_power, norm_dist = (list(t) for t in zip(*sorted(zip(omega, norm_power, norm_dist))))

		# Plot distance as a fct of power in a loglog graph for different omega values
		fig, ax = Plot.initPlot()
		for i in range(len(set(omega_sorted))):
			ax.loglog(norm_power[n_omega*i:n_omega*i+n_omega], norm_dist[n_omega*i:n_omega*i+n_omega], \
				'.-', label="$\omega = $ " + num2str(omega_sorted[n_omega*i]/omega_res[i]) + " $\omega_{res}$")
		for powerEff in np.logspace(-5, 5, 30):
			x = np.array([5e-5, 2])
			plt.plot(x, powerEff *x, 'k--', alpha = 0.5)
		plt.title("Pareto curves for " + str(len(self.y[0])) + " iterations " + opt_type + \
			" optimizations with " + num2str(sim_time) + "s simulations")
		Plot.configurePlot(fig, ax, 'Relative Power','Relative Speed', legendLocation='lower right', size='small')
		ax.set_xlim([5e-5, 2])
		ax.set_ylim([1e-3, 5])
		if show: plt.show()
		if save: plt.savefig(filename + ".png", format='png', dpi=300)
