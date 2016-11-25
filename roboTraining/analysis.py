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
# plt.style.use('fivethirtyeight')
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
plt.rc('axes', facecolor='white')
plt.rc('savefig', facecolor='white')

class Analysis(object):

	def __init__(self, root="/home/gabs48/edu/Data", folder="test_uniformite"):
		"""Init the analysis class with a folder containing a set of results"""

		self.root = root
		self.folder = folder
		self.path = os.path.join(root, folder)

		self.scores = []
		self.dists = []
		self.pows = []
		self.filenames = []
		self.parameters = []
		self.trainable = []
		self.params_names = []
		self.opt_type = []
		self.sim_time = []
		self.omega = []
		self.ts = []
		self.sl = []
		self.k = []
		self.m = []
		self.n_nodes = []
		self.n_conns = []
		self.n_springs = []
		self.ps = []
		self.ref_pow = []
		self.ref_dist = []

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

		self.y_d = []
		self.y_d_av = []
		self.y_d_std_a = []
		self.y_d_std_b = []
		self.y_d_std = []
		self.y_d_max = []
		self.y_d_min = []

		self.y_p = []
		self.y_p_av = []
		self.y_p_std_a = []
		self.y_p_std_b = []
		self.y_p_std = []
		self.y_p_max = []
		self.y_p_min = []

		self.y_sce = []
		self.y_sce_f = []
		self.pc1 = []
		self.pc2 = []
		self.pc1_av = []
		self.pc2_av = []
		self.pc1_std = []
		self.pc2_std = []
		self.window = []

		self.pcaMat = None

	def _load_parameters(self):
		"""Load the training parameters from all parameters files"""

		for i, f in enumerate(self.filenames):
			with open(f.replace("score", "parameter"), 'r') as csvfile:
				tab = list(csv.reader(csvfile, delimiter=';', quotechar='|'))
			
				params = []
				names = []

				# ## Can read params.csv version 1
				# mult = len(self.trainable[i]) + 2
				# for j in range(len(tab)) :
				# 	if j%mult == 2:
				# 		row = tab[j]
				# 	if j%mult in range(3, mult-1):
				# 		row.extend(tab[j])
				# 	if j%mult == 0 and j != 0:
				# 		row_float = list(map(float, row))
				# 		params.append(row_float)

				### Can read params.csv version 2
				mult = len(self.trainable[i]) + 1
				for j in range(len(tab)) :
					if j%mult == 1:
						if j == 1:
							names.append(str(tab[j][1]))
						row = tab[j][2:]
					if j%mult in range(2, mult):
						if j in range(2, mult):
							names.append(str(tab[j][1]))
						row.extend(tab[j][2:])
					if j%mult == 0 and j != 0:
						row_float = list(map(float, row))
						params.append(row_float)


				row_float = list(map(float, row))
				params.append(row_float) # last iteration
				self.params_names.append(names)
				self.parameters.append(params)
				print(" -- Parameters of " + str(i+1) + "/" + str(len(self.filenames)) + " loaded -- ")

	def _load_configs(self):
		"""Load configuration parameters from all the config files"""

		for f in self.filenames:
			with open(f.replace("score", "config"), 'r') as csvfile:
				tab = list(csv.reader(csvfile, delimiter=';', quotechar='|'))

				# Find node and connection numbers
				nn_ind = findIndex(tab, "noNodes:")
				n_nodes = int(tab[nn_ind[0]][nn_ind[1] + 1])
				self.n_nodes.append(n_nodes)
				nc_ind = findIndex(tab, "noNeighbours:")
				n_conns = int(tab[nc_ind[0]][nc_ind[1] + 1])
				self.n_conns.append(n_conns)
				self.n_springs.append((n_nodes - 1 - (n_conns - 1) / 2) * n_conns)


				# Find trainable parameters
				params_ind = findIndex(tab, "trainableParams:")
				vals = []
				for i in range(len(tab) - params_ind[0] + 1):
					if len(tab[i]) > (params_ind[1] + 2):
						if str(tab[i][params_ind[1] + 2]).find("name") != -1:
							dico = dict()
							dico["name"] = str(tab[i][params_ind[1] + 3]).translate(None, ' :')
							dico["min"] = float(tab[i + 1][params_ind[1] + 3])
							dico["max"] = float(tab[i + 2][params_ind[1] + 3])
							if dico["name"].find("base_") == -1:
								dico["num"] = (n_nodes - 1 - (n_conns - 1) / 2) * n_conns
							else:
								dico["num"] = 1
							vals.append(dico)
				self.trainable.append(vals)

				# Find default omega
				omega_ind = findIndex(tab, "omega:")
				self.omega.append(float(tab[omega_ind[0] + 1][omega_ind[1] + 2]))

				# Find the optimization type
				rand_type_ind = findIndex(tab, "noInstances:")
				cma_type_ind = findIndex(tab, "maxIter:")
				if rand_type_ind != [-1, -1]:
					self.opt_type.append("RANDOM")
				elif cma_type_ind != [-1, -1]:
					self.opt_type.append("CMA")
				else:
					self.opt_type.append("GENETIC")

				# Find simulation time
				ts_ind = findIndex(tab, "timeStep:")
				sl_ind = findIndex(tab, "simulationLength:")
				self.ts.append(float(tab[ts_ind[0]][ts_ind[1] + 1]))
				self.sl.append(int(tab[sl_ind[0]][sl_ind[1] + 1]))
				self.sim_time.append(float(tab[ts_ind[0]][ts_ind[1] + 1])*float(tab[sl_ind[0]][sl_ind[1] + 1]))

				# Find population size
				if self.opt_type[-1] == "CMA":
					p = 0
					for k in range(len(vals)):
						p +=  vals[k]["num"]
					self.ps.append(int(4 + math.floor(3 * math.log(p))))
				elif self.opt_type[-1] == "GENETIC":
					ps_ind = findIndex(tab, "populationSize:")
					self.ps.append(float(tab[ps_ind[0]][ps_ind[1] + 1]))
				else:
					self.ps.append(1)

				# Find reference distances and powers
				ref_pow_ind = findIndex(tab, "refPower:")
				ref_dist_ind = findIndex(tab, "refDist:")
				if ref_pow_ind != [-1, -1]:
					self.ref_pow.append(float(tab[ref_pow_ind[0]][ref_pow_ind[1] + 1]))
				else:
					self.ref_pow.append(-1)
				if ref_dist_ind != [-1, -1]:
					self.ref_dist.append(float(tab[ref_dist_ind[0]][ref_dist_ind[1] + 1]))
				else:
					self.ref_dist.append(-1)

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
					if window == 0:
						window = 1
				y_av = np.convolve(np.array(y), np.ones((window,))/window, mode='valid')
				y_d_av = np.convolve(np.array(self.y_d[i]), np.ones((window,))/window, mode='valid')
				y_p_av = np.convolve(np.array(self.y_p[i]), np.ones((window,))/window, mode='valid')
				if window%2 == 0:
					x_av = self.x[i][window/2:len(self.x[i])-window/2+1]
				else:
					x_av = self.x[i][window/2:len(self.x[i])-window/2]

				# Compute std deviation
				y_std = self._window_stdev(np.array(y), window=window)
				y_d_std = self._window_stdev(np.array(self.y_d[i]), window=window)
				y_p_std = self._window_stdev(np.array(self.y_p[i]), window=window)
				x_std = self.x[i][window:len(self.x[i])-window+1]
				if window%2 == 0:
					y_std_a = y_av[window/2:len(y_av)-window/2] + y_std
					y_std_b = y_av[window/2:len(y_av)-window/2] - y_std
					y_d_std_a = y_d_av[window/2:len(y_d_av)-window/2] + y_d_std
					y_d_std_b = y_d_av[window/2:len(y_d_av)-window/2] - y_d_std
					y_p_std_a = y_p_av[window/2:len(y_p_av)-window/2] + y_p_std
					y_p_std_b = y_p_av[window/2:len(y_p_av)-window/2] - y_p_std
				else:
					y_std_a = y_av[window/2:len(y_av)-window/2-1] + y_std
					y_std_b = y_av[window/2:len(y_av)-window/2-1] - y_std
					y_d_std_a = y_d_av[window/2:len(y_d_av)-window/2-1] + y_d_std
					y_d_std_b = y_d_av[window/2:len(y_d_av)-window/2-1] - y_d_std
					y_p_std_a = y_p_av[window/2:len(y_p_av)-window/2-1] + y_p_std
					y_p_std_b = y_p_av[window/2:len(y_p_av)-window/2-1] - y_p_std

				# Compute min and max
				val_s_max = y[0]
				val_s_min = y[0]
				val_d_max = y[0]
				val_d_min = y[0]
				val_p_max = y[0]
				val_p_min = y[0]
				y_max = []
				y_min = []
				for val in y:
					if val_s_max < val:
						val_s_max = val
					if val_s_min > val:
						val_s_min = val
					y_max.append(val_s_max)
					y_min.append(val_s_min)
				y_d_max = []
				y_d_min = []
				for val in self.y_d[i]:
					if val_d_max < val:
						val_d_max = val
					if val_d_min > val:
						val_d_min = val
					y_d_max.append(val_d_max)
					y_d_min.append(val_d_min)
				y_p_max = []
				y_p_min = []
				for val in self.y_p[i]:
					if val_p_max < val:
						val_p_max = val
					if val_p_min > val:
						val_p_min = val
					y_p_max.append(val_p_max)
					y_p_min.append(val_p_min)

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
						pc1_std = pc1_av[window/2:len(pc1_av)-window/2-1]
						pc2_std = pc2_av[window/2:len(pc2_av)-window/2-1]
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
				self.x_av.append(x_av)
				self.x_std.append(x_std)

				self.y_av.append(y_av)
				self.y_std_a.append(y_std_a)
				self.y_std_b.append(y_std_b)
				self.y_std.append(y_std)
				self.y_max.append(y_max)
				self.y_min.append(y_min)

				self.y_d_av.append(y_d_av)
				self.y_d_std_a.append(y_d_std_a)
				self.y_d_std_b.append(y_d_std_b)
				self.y_d_std.append(y_d_std)
				self.y_d_max.append(y_d_max)
				self.y_d_min.append(y_d_min)

				self.y_p_av.append(y_p_av)
				self.y_p_std_a.append(y_p_std_a)
				self.y_p_std_b.append(y_p_std_b)
				self.y_p_std.append(y_p_std)
				self.y_p_max.append(y_p_max)
				self.y_p_min.append(y_p_min)

				self.y_sce.append(y_sce)
				self.y_sce_f.append(y_sce_f)
				self.x_sce_f.append(x_sce_f)

				i += 1

	def _rearrange_pop(self, index=0, unit="score"):
		"""Group a score list in generations and return gen number min, max and average"""


		dim = len(self.y[index])
		ps = self.ps[index]
		assert dim%ps == 0, "The total number of iteration (" + str(dim) + ") shall be a " + \
			"multiple of the population size (" + str(ps) + "). Please verify the file " +  \
			self.filenames[index] + " or this sript!"

		array = np.array(self.y[index])
		if unit == "distance":
			array = np.array(self.y_d)
		if unit == "power":
			array = np.array(self.y_p)
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

	def _get_style_colors(self):
		""" Return a arry with the current style colors """

		if 'axes.prop_cycle' in plt.rcParams:
			cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
		else:
			cols = ['b', 'r', 'y', 'g', 'k']
		return cols

	def load(self):
		"""Browse all folder and retrieve all scores. Then load configs and parameters"""

		# Load scores
		for path, subdirs, files in os.walk(self.path):
			for name in files:
				if name.find("score") != -1 and os.path.splitext(name)[1] == ".csv":
					with open(os.path.join(path, name), 'r') as csvfile:
						tab = csv.reader(csvfile, delimiter=';', quotechar='|')
						self.filenames.append(os.path.join(path, name))
						i = 0
						for row in tab:
							if i == 0:
								self.scores.append(row)
							elif i == 1:
								self.pows.append(row)
							else:
								self.dists.append(row)
							i += 1

		# Fill score list and iteration range
		i = 0
		for y in self.scores:
			self.y.append(list(map(float, y)))
			if i < len(self.dists):
				self.y_d.append(list(map(float, self.dists[i])))
			if i < len(self.pows):
				self.y_p.append(list(map(float, self.pows[i])))
			self.x.append(range((len(y))))
			i += 1

		# Load configs and parameters
		self._load_configs()

	def plot_raw(self, index=0, filename="results_raw", unit="score", title=None, show=False, save=True):
		"""Plot score evolution for a given file"""
		
		print(" -- Printing raw " + unit + " bar graph for file " + self.filenames[index])

		val = self.y
		if unit == "distance":
			val = self.y_d
		if unit == "power":
			val = self.y_p
		fig, ax = Plot.initPlot()
		for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
			item.set_fontsize(17)

		barWidth = 0.35
		opacity = 0.4
		plt.bar(self.x[index], val[index], barWidth, alpha = opacity, color = 'k', label = 'distance traveled')
		if title != None:
			plt.title(title)
		else:
			plt.title("Training " + unit + " (optimum  = " + str(max(val[index])) + ")")
		Plot.configurePlot(fig, ax, 'Temp', 'Temp', legend = True, legendLocation = 'lower center')
		plt.xlabel('Iteration')
		plt.ylabel(unit)
		if show: plt.show()
		if save: plt.savefig(filename + "_" + unit + ".png", format='png', dpi=300)
		plt.close()

	def plot_gen(self, index=0, filename="results_gen", unit="score", title=None, show=False, save=True):
		"""Plot scores rearranged in generations for a given file"""

		if self.opt_type[index] != "RANDOM":
			x, y_min, y_max, y_av = self._rearrange_pop(index, unit)

			print(" -- Printing generation " + unit + " for file " + self.filenames[index])

			fig, ax = Plot.initPlot()
			for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
				item.set_fontsize(17)

			plt.plot(x, y_max, linestyle="-", color=self._get_style_colors()[1], linewidth=0.5, label="Max")
			plt.plot(x, y_av, linestyle="-", color=self._get_style_colors()[3], linewidth=0.5, label="Average")
			plt.plot(x, y_min, linestyle="-", color=self._get_style_colors()[0], linewidth=0.5, label="Min")
			if title != None:
				plt.title(title)
			else:
				plt.title(self.filenames[index])#"Training " + unit + " of " + self.opt_type[index] + " algorithm with popSize = " + \
				#num2str(self.ps[index]))
			Plot.configurePlot(fig, ax, 'Temp', 'Temp', legend=True, legendLocation='lower center')
			plt.xlabel('Generation Epoch')
			plt.ylabel(unit.title())
			plt.xlim([0, max(x)])
			if show: plt.show()
			if save: plt.savefig(filename + "_" + unit + ".png", format='png', dpi=300)
			plt.close()

		else:
			print(" -- Can't print generation " + unit + " for file " + self.filenames[index] + \
				" with "+ self.opt_type[index] + " optimization type.")

	def plot_raw_av(self, index=0, filename="results_raw_av", unit="score", title=None, show=False, save=True):
		"""Plot average and max score evolution for a given file"""

		if not self.y_max:
			self._compute_stats()

		val = self.y
		val_max = self.y_max
		val_std_a = self.y_std_a
		val_av = self.y_av
		val_std_b = self.y_std_b
		val_min = self.y_min
		if unit == "distance":
			val = self.y_d
			val_max = self.y_d_max
			val_std_a = self.y_d_std_a
			val_av = self.y_d_av
			val_std_b = self.y_d_std_b
			val_min = self.y_d_min
		if unit == "power":
			val = self.y_p
			val_max = self.y_p_max
			val_std_a = self.y_p_std_a
			val_av = self.y_p_av
			val_std_b = self.y_p_std_b
			val_min = self.y_p_min

		print(" -- Printing " + unit + " stats graph for file " + self.filenames[index])

		fig, ax = Plot.initPlot()
		for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
			item.set_fontsize(17)

		plt.plot(self.x[index], self.y_max[index] ,"r-", label="maximum")
		plt.plot(self.x_std[index], val_std_a[index] ,"r--", linewidth=0.2, label="max std dev")
		plt.plot(self.x_av[index], val_av[index], "g-", label="average score")
		plt.plot(self.x_std[index], val_std_b[index] ,"b--", linewidth=0.2, label="min std dev")
		plt.plot(self.x[index], val_min[index], "b-", label="minimum")
		if title != None:
			plt.title(title)
		else:
			plt.title("Training max and average " + unit + " (optimum  = " + str(max(self.y[index])) + ")")
		Plot.configurePlot(fig, ax, 'Temp', 'Temp', legend=True, legendLocation='lower center')
		plt.xlabel('Iteration')
		plt.ylabel(unit)
		if show: plt.show()
		if save: plt.savefig(filename + "_" + unit + ".png", format='png', dpi=300)
		plt.close()

	def plot_conv_err(self, index=0, filename="results_score_conv_err", title=None, show=False, save=True):
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

	def plot_noise_sim(self, index=0, filename="results_noise_sim", title=None, nPoints=5, window=2, show=False, save=True):
		"""Perform simulations for different values of noise with the best individu of a file 
		and plot the results"""

		print(" -- Printing noise simulation graph for file " + self.filenames[index])

		noiseArr = np.logspace(-8, -1, num=nPoints)
		if window%2 == 0:
			noiseArr_av = noiseArr[window/2:nPoints-window/2+1]
		else:
			noiseArr_av = noiseArr[window/2:nPoints-window/2]
		distArr = np.zeros(nPoints)
		powerArr = np.zeros(nPoints)

		for i in range(nPoints):
			noise = noiseArr[i]
			print(" -- Simulation " + str(i+1) + "/" + str(nPoints) + " : simulation noise value = " + str(noise) + " -- ")
			bestIndex = np.argmax(self.y[index])
			[score, distArr[i], powerArr[i]] = self.simulate_ind(index, bestIndex, movie=False, rc=False, simTime=100, \
				simNoise=noise)

		averageArr = np.convolve(np.array(distArr), np.ones((window,))/window, mode='valid')
		fig, ax = Plot.initPlot()
		ax.semilogx(noiseArr, distArr, 'r.', label = "Noisy simulation scores" )
		ax.semilogx(noiseArr_av, averageArr, 'b-', label = "Score average")
		plt.title("Simulation accuracy with increasing simulation relative noise")
		Plot.configurePlot(fig, ax, "Relative noise" + r'$ \ \sigma$ on simulation step values', "Distance Traveled [m]", legend = False)
		if show: plt.show()
		if save: plt.savefig(filename + ".png", format='png', dpi=300)
		plt.close()

	def plot_noise_params(self, index=0, filename="results_noise_params", title=None, nPoints=75, window=15, show=False, save=True):
		"""Perform simulations for different values of noise on the parameters with the best individu of a file 
		and plot the results"""

		print(" -- Printing noise parameters graph for file " + self.filenames[index])

		noiseArr = np.logspace(-8, 0, num=nPoints)
		if window%2 == 0:
			noiseArr_av = noiseArr[window/2:nPoints-window/2+1]
		else:
			noiseArr_av = noiseArr[window/2:nPoints-window/2]
		distArr = np.zeros(nPoints)
		powerArr = np.zeros(nPoints)

		for i in range(nPoints):
			noise = noiseArr[i]
			print(" -- Simulation " + str(i+1) + "/" + str(nPoints) + " : parameters noise value = " + str(noise) + " -- ")
			bestIndex = np.argmax(self.y[index])
			[score, distArr[i], powerArr[i]] = self.simulate_ind(index, bestIndex, movie=False, rc=False, paramNoise=noise)

		averageArr = np.convolve(np.array(distArr), np.ones((window,))/window, mode='valid')
		fig, ax = Plot.initPlot()
		ax.semilogx(noiseArr, distArr, 'r.', label = "Noisy parameters scores" )
		ax.semilogx(noiseArr_av, averageArr, 'b-', label = "Score average")
		plt.title("Simulation accuracy with increasing parameters relative noise")
		Plot.configurePlot(fig, ax, "Relative noise" + r'$ \ \sigma$ on individu parameters', "Distance Traveled [m]", legend = False)
		if show: plt.show()
		if save: plt.savefig(filename + ".png", format='png', dpi=300)
		plt.close()

	def plot_param(self, index=0, filename="evol_param", paramIndex=0, title=None, show=False, save=True):
		"""Plot score evolution for a given file"""
		
		if not self.parameters:
			self._load_parameters()

		print(" -- Printing param " + str(paramIndex) + " graph for file " + self.filenames[index])

		p = []
		for i, row in enumerate(self.parameters[index]):
			if i != 0:
				p.append(row[paramIndex])
		plt.plot(self.x[index], p, color='b')
		if title != None:
			plt.title(title)
		else:
			plt.title("Evolution of param " + str(paramIndex) + " (optimum  = " + str(max(self.y[index])) + ")")
		plt.xlabel('Iteration')
		plt.ylabel("Parameter " + str(paramIndex))

		if show: plt.show()
		if save: plt.savefig(filename + "_" + str(paramIndex) + ".png", format='png', dpi=300)
		plt.close()

	def plot_all_raws(self, filename="results_raw", unit="score", show=False, save=True):
		"""Plot score evolution for a all files. This can take several minutes"""

		i = 0
		for y in self.y:
			self.plot_raw(index=i, filename=filename + "_" + str(i), unit=unit, show=show, save=save)
			i += 1

	def plot_all_gens(self, filename="results_gen", unit="score", show=False, save=True):
		"""Plot score evolution rearranged in generations for all files."""

		i = 0
		for y in self.y:
			self.plot_gen(index=i, filename=filename + "_" + str(i), unit=unit, show=show, save=save)
			i += 1

	def plot_all_raws_av(self, filename="results_raw_av", unit="score", show=False, save=True):
		"""Plot average and max score evolution for a all files."""

		i = 0
		for y in self.y:
			self.plot_raw_av(index=i, filename=filename + "_" + str(i), unit=unit, show=show, save=save)
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

	def plot_all_noise_sims(self, filename="results_noise_sim", show=False, save=True):
		"""Plot accuracy evoluation in noisy simulation for all files of the current folder"""

		print(" -- Printing noise simulation evolution for each file. This can take a while -- ")

		i = 0
		for y in self.y:
			self.plot_noise_sim(index=i, filename=filename + "_" + str(i), show=show, save=save)
			i += 1

	def plot_all_noise_params(self, filename="results_noise_params", show=False, save=True):
		"""Plot accuracy evoluation when loading noisy parameters for all files of the current folder"""

		print(" -- Printing noise parameters evolution for each file. This can take a while -- ")

		i = 0
		for y in self.y:
			self.plot_noise_params(index=i, filename=filename + "_" + str(i), show=show, save=save)
			i += 1

	def plot_all_params(self, filename="param_", show=False, save=True):

		if not self.parameters:
			self._load_parameters()

		print(" -- Printing parameters evolution for each file. This can take a while -- ")

		i = 0
		for p in self.parameters[0]:
			self.plot_param(index=0, paramIndex=i, filename=filename + "_" + str(i), show=show, save=save)
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

	def simulate_ind(self, index1=None, index2=None, simTime=None, simName="Simulation", rc=False, \
		movie=True, pca=False, transPhase=0, trainingPhase=0.9, openPhase=0.1, alpha=0.0001, beta=1, \
		pcaTitle="PCA", pcaFilename="pca", nrmse=False, simNoise=0, paramNoise=0, noiseType="rand"):
		"""Render a simulation movie for a given individu"""

		# Init variables
		scoreFilename = self.filenames[index1]
		if not simTime:
			sl = self.sl[index1]
		else:
			sl = int(simTime/self.ts[index1])
		simTime = self.ts[index1] * sl

		print(' -- Simulate individu ' +  scoreFilename + ' (' + str(index2) + "/" + \
			str(len(self.scores[index1])) + ") with score {:.4f}".format(self.y[index1][index2]) + \
			" for " + num2str(simTime) + "s. --")

		# Construct robot from config file
		configFilename = scoreFilename.replace("score", "config")
		env = HardEnvironment()
		morph = SpringMorphology(noNodes=self.n_nodes[index1] , spring=self.k[index1], noNeighbours=3, environment=env)
		control = ClosedLoopSineControl(morph)
		control.loadCSV(configFilename)
		robot = Robot(morph, control)

		# Get training parameters of a given individu and update them in the robot
		parameterFilename = scoreFilename.replace("score", "parameter")
		trainscheme = TrainingScheme()
		for param in self.trainable[index1]:
			trainscheme.createTrainVariable(param["name"], param["min"], param["max"])
		

		# File vs 1
		# for param in self.trainable[index1]:
		# 	print param
		# 	trainscheme.createTrainVariable(param["name"], param["min"], param["max"])
		# paramMatrix = trainscheme.loadCSV(parameterFilename, index2, len(self.trainable[index1]))
		# if paramNoise != 0:
		# 	paramMatrix = paramMatrix + np.random.standard_normal(paramMatrix.shape) * paramNoise
		# 	l_vals = paramMatrix < 0.0
		# 	h_vals = paramMatrix > 1.0
		# 	paramMatrix[l_vals] = 0.0
		# 	paramMatrix[h_vals] = 1.0
		# trainscheme.normalizedMatrix2robot(paramMatrix, robot)

		# File vs 2
		if not self.parameters:
			self._load_parameters()
		# Reorganise the parameters in the rigt order
		rank = []
		for param in self.params_names[index1]:
			for i, p in enumerate(self.trainable[index1]):
				if param == p["name"]:
					rank.append(i)
					break;
		paramList = trainscheme.loadCSV2List(parameterFilename, index2, rank)
		trainscheme.normalizedList2robot(paramList, robot)

		# Create the simulation
		plotter = Plotter(movie=movie, plot=movie, movieName=simName, plotCycle = 6)
		simulEnv = SimulationEnvironment(timeStep=self.ts[index1], simulationLength=sl, plot=plotter,\
			pcaPlot=pca, pcaTitle=pcaTitle, pcaFilename=pcaFilename, pcaMat=self.pcaMat, \
			perfMetr="powereff", controlPlot=False, refPower=self.ref_pow[index1], refDist=self.ref_dist[index1])

		# Do the simulation
		if rc:
			if nrmse:
				simul = ForceTrainingSimulation(simulEnv, robot, \
					transPhase=transPhase, trainPhase=trainingPhase, openPhase=openPhase, \
					trainingPlot="cont", alpha=alpha, beta=beta, outputFilename="control_n_" + \
					str(self.n_nodes[index1]), outputFolder="nodes_CL_pic")
			else:
				#simul = TrainedSimulation(simulEnv, robot, filename="w.pkl")
				simul = ForceTrainingSimulation(simulEnv, robot, \
					transPhase=transPhase, trainPhase=trainingPhase, openPhase=openPhase, \
					trainingPlot="all", alpha=alpha, beta=beta, outputFilename="training", \
					outputFolder="ResLearning_" + str(transPhase) + "_" +  str(trainingPhase) + "_" + \
					str(openPhase) + "_" + str(alpha) + "_" + str(beta))
		else:
			if simNoise !=  0:
				if noiseType == "impulse":
					simul = NoisyImpulseVerletSimulation(simulEnv, robot, noise=simNoise)
				elif noiseType == "rand":
					simul = NoisyVerletSimulation(simulEnv, robot, noise=simNoise)
			else:
				simul = VerletSimulation(simulEnv, robot)
		[score, power, distance] = simul.runSimulation();

		print(" -- Simulation terminated with score {:.4f}".format(score) + \
			". Distance:  {:.2f}".format(distance) + " and Power:  {:.2f}".format(power) + " -- ")
		if movie:
			print(" -- Video saved in file " + simName + ".mp4 --")
		if pca and self.pcaMat == None:
			self.pcaMat = simulEnv.pcaMat

		if nrmse:
			return [score, simul.getDistance(), robot.getPower(), simul.get_training_error()]
		else:
			return [score, simul.getDistance(), robot.getPower()]

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
			evo_av_title = "Averaged optimization evolution with " + self.opt_type[i] + " training and " + \
				num2str(self.sim_time[i]) + " s simulations"
			evo_title = "Otimization evolution with " + self.opt_type[i] + " training and " + \
				num2str(self.sim_time[i]) + " s simulations"
			gen_title = "Generation evolution with " + self.opt_type[i] + " training and " + \
				num2str(self.sim_time[i]) + " s simulations"
			ss_title = "PC parameters evolution with " + self.opt_type[i] + " training and " + \
				num2str(self.sim_time[i]) + " s simulations"
			err_filename = folder + "err_" + ext
			evo_filename = folder + "evol_" + ext
			evo_av_filename = folder + "av_evol_" + ext
			gen_filename = folder + "gen_" + ext
			ss_filename = folder + "exp_" + ext
			self.plot_score(index=i, filename=evo_filename, title=evo_title)
			self.plot_score_av(index=i, filename=evo_av_filename, title=evo_av_title)
			self.plot_conv_err(index=i, filename=err_filename, title=err_title)
			self.plot_gen(index=i, filename=gen_filename, title=gen_title)
			self.plot_state_space(index=i, filename=ss_filename, title=ss_title)
			i += 1

	def km(self, filename="results_km", sensibility=False, show=False, save=True):
		"""Perform specific analysis for a km batch"""

		folder = "km_pic/"
		mkdir_p(folder)

		
		print(" -- Mass-spring Analysis of folder " + self.path + " --")

		k = []
		m = []
		d = []
		sim_time = self.sim_time[0]
		opt_type = self.opt_type[0]

		# Fill values from loaded variables
		for i in range(len(self.y)):

			duplicate = False
			assert self.sim_time[i] == sim_time, \
				"For a meaningfull pareto graph, ensure the simulation times are the same for all data"
			assert self.opt_type[i] == opt_type, \
				"For a meaningfull pareto graph, ensure the optimization algorithms are the same for all data"

			# Fetch iteration k and m value
			it_k = self.k[i]
			it_m = self.m[i]

			# If couple already existsn average with previous one
			for j, spring in enumerate(k):
				for l, mass in enumerate(m):
					if spring == it_k and \
						mass == it_m and j == l:
							d[j].append(self.get_best_ind(index=i)[0])
							duplicate = True

			if duplicate == False:
				d.append([self.get_best_ind(index=i)[0]])
				k.append(it_k)
				m.append(it_m)

		# Average points with multiple values
		n_av = []
		for i in range(len(d)):
			if  len(d[i]) > 1:
				n_av.append(len(d[i]))
			d[i] = sum(d[i]) / len(d[i])
		if len(n_av) != 0:
			print(" -- Averaging " + str(len(n_av)) + " graph points with on average " + \
				num2str(float(sum(n_av)/len(n_av))) + " data sets for each --")

		# Sort lists
		m, k, d = (list(t) for t in zip(*sorted(zip(m, k, d))))
		c = []
		for a in k:
			c.append(float(1.0) / a)
		n_k = len(k) / len(set(k))
		n_m = len(m) / len(set(m))

		# Plot distance as a fct of k in a graph for different m values
		col = ["b-", "g-", "r-", "c-", "m-", "y-"]
		fig, ax = Plot.initPlot()
		for i in range(len(set(m))):
			c_res = 1 / (25 * m[n_m*i])
			ax.plot(c[n_m*i:n_m*i+n_m], d[n_m*i:n_m*i+n_m], col[i%len(col)], label="m = " + num2str(m[n_m*i]) + " kg")
			ax.plot([c_res, c_res], [0, 0.6], col[i%len(col)] + "-")
		plt.title(" Maximum scores in fct of compliance for " + str(len(self.y[0])) + " iterations " + opt_type + \
			" optimizations with " + num2str(sim_time) + "s simulations")
		Plot.configurePlot(fig, ax, 'Spring Compliance ($1/k$)','Maximum score', legendLocation='lower right', size='small')
		ax.set_xlim([0, 0.06])
		ax.set_ylim([0, 0.6])
		if show: plt.show()
		if save: plt.savefig(folder + filename + ".png", format='png', dpi=300)

		if sensibility:
			print "Il faut faire ca!"

	def nodes(self, filename="results_nodes", noiseAnalysis=False, videoAnalysis=False, \
		genAnalysis=True, show=False, save=True):
		"""Perform specific analysis for a nodes batch"""

		folder = "nodes_pic/"
		mkdir_p(folder)

		nodes = []
		dist = []
		power = []
		score = []
		score_test = []
		sim_time = self.sim_time[0]
		opt_type = self.opt_type[0]

		# Fill values from loaded variables
		for i in range(len(self.y)):

			duplicate = False
			assert self.sim_time[i] == sim_time, \
				"For a meaningfull graph, ensure the simulation times are the same for all data"
			assert self.opt_type[i] == opt_type, \
				"For a meaningfull graph, ensure the optimization algorithms are the same for all data"

			# Fetch iteration nodes number value and best individu
			it_nodes = self.n_nodes[i]
			best = self.get_best_ind(index=i)

			# If nodes number already existsn average with previous one
			# print i, nodes, it_nodes
			for j, no in enumerate(nodes):
				if no[0] == it_nodes:
					s, p, d = self.simulate_ind(best[1], best[2], simTime=10, movie=False)
					score_test[j].append(s)
					dist[j].append(self.y_d[best[1]][best[2]])
					power[j].append(self.y_p[best[1]][best[2]])
					score[j].append(best[0])
					nodes[j].append(it_nodes)
					duplicate = True

			if duplicate == False:
				s, p, d = self.simulate_ind(best[1], best[2], simTime=10, movie=False)
				score_test.append([s])
				dist.append([self.y_d[best[1]][best[2]]])
				power.append([self.y_p[best[1]][best[2]]])
				score.append([best[0]])
				nodes.append([it_nodes])
				if noiseAnalysis:
					print " -- Producting noise analysis graphs for " + str(it_nodes) + " nodes --"
					self.plot_noise_sim(best[1], filename=folder + "noise_n_" + str(it_nodes),\
						title="N = " + str(it_nodes), nPoints=60, window=10)
				if videoAnalysis:
					if it_nodes == 20:
						print " -- Producting simulation video with " + str(it_nodes) + " nodes --"
						score,self.simulate_ind(best[1], best[2], simTime=10, movie=True, rc=False, simName="Sim_" + \
							str(it_nodes))
				if genAnalysis:
					print " -- Producting generation graphs for " + str(it_nodes) + " nodes --"
					self.plot_gen(best[1], filename=folder + "gen_n_" + str(it_nodes), \
						title="CMA-ES evolution of " + str(it_nodes) + " nodes structure")


		# Average points with multiple values
		n_av = 0
		dist_std = np.zeros(len(dist))
		power_std = np.zeros(len(power))
		score_std = np.zeros(len(score))
		robust = np.zeros(len(score))
		robust_std = np.zeros(len(score))
		for i in range(len(dist)):
		 	n_av += len(dist[i])
		 	dist_std[i] = np.std(np.array(dist[i]))
		 	power_std[i] = np.std(np.array(power[i]))
		 	score_std[i] = np.std(np.array(score[i]))
		 	robust_std[i] = np.std(np.array(score[i]) - np.array(score_test[i]))
		 	dist[i] = sum(dist[i]) / len(dist[i])
		 	power[i] = sum(power[i]) / len(power[i])
		 	robust[i] = np.mean(np.array(score[i]) - np.array(score_test[i]))
		 	score[i] = sum(score[i]) / len(score[i])
			nodes[i] = sum(nodes[i]) / len(nodes[i])

		if n_av != len(dist):
			print(" -- Averaging " + str(n_av) + " graph points with on average " + \
				num2str(float(n_av)/len(dist)) + " data sets for each --")

		# Sort lists
		nodes, dist, power, score, robust, dist_std, power_std, score_std, robust_std= \
			(list(t) for t in zip(*sorted(zip(nodes, dist, power, score, robust, dist_std, power_std, score_std, robust_std))))

		# Plot distance and power as a fonction of the nodes number
		fig, ax = Plot.initPlot()
		ax.errorbar(nodes, dist, \
			yerr=dist_std, fmt='.-', ecolor='r', \
			linewidth=1.5, label="Distance")
		plt.title("Travelled distance in 10s in function of nodes number")
		Plot.configurePlot(fig, ax, 'Nodes number','Distance', legendLocation='lower right', size='small')
		if show: plt.show()
		if save:
			print(" -- Print distance evolution with nodes number in " + folder + filename + "_dist.png --")
			plt.savefig(folder + filename + "_dist.png", format='png', dpi=300)

		fig, ax = Plot.initPlot()
		ax.errorbar(nodes, power, \
			yerr=power_std, fmt='.-', ecolor='r', \
			linewidth=1.5, label="Power") 
		plt.title("Dissipated power in function of nodes number")
		Plot.configurePlot(fig, ax, 'Nodes number','Power', legendLocation='lower right', size='small')
		if show: plt.show()
		if save:
			print(" -- Print power evolution with nodes number in " + folder + filename + "_power.png --")
			plt.savefig(folder + filename + "_power.png", format='png', dpi=300)

		fig, ax = Plot.initPlot()
		ax.errorbar(nodes, score, \
			yerr=score_std, fmt='.-', ecolor='r', \
			linewidth=1.5, label="Score")
		plt.title("Best individu score in function of nodes number")
		Plot.configurePlot(fig, ax, 'Nodes number','Score', legendLocation='lower right', size='small')
		if show: plt.show()
		if save:
			print(" -- Print score evolution with nodes number in " + folder + filename + "_score.png --")
			plt.savefig(folder + filename + "_score.png", format='png', dpi=300)

		fig, ax = Plot.initPlot()
		ax.errorbar(nodes, robust, \
			yerr=robust_std, fmt='.-', ecolor='r', \
			linewidth=1.5, label="Accuraccy") 
		plt.title("Difference of score between noisy and straight simulations")
		Plot.configurePlot(fig, ax, 'Nodes number','Score difference', legendLocation='lower right', size='small')
		if show: plt.show()
		if save:
			print(" -- Print score differences with nodes number in " + folder + filename + "_score_diff.png --")
			plt.savefig(folder + filename + "_score_diff.png", format='png', dpi=300)

	def freq(self, filename="results_freq", show=False, save=True):
		"""Perform specific analysis for a omega batch"""

		folder = "omega_pic/"
		mkdir_p(folder)
		print(" -- Omega Analysis of folder " + self.path + " --")

		score = []
		omega = []
		nodes = []
		sim_time = self.sim_time[0]
		opt_type = self.opt_type[0]

		# Fill values from loaded variables
		for i in range(len(self.y)):

			duplicate = False
			assert self.sim_time[i] == sim_time, \
				"For a meaningfull pareto graph, ensure the simulation times are the same for all data"
			assert self.opt_type[i] == opt_type, \
				"For a meaningfull pareto graph, ensure the optimization algorithms are the same for all data"

			# Fetch iteration omega value
			it_omega = None
			for it in self.trainable[i]:
				if it["name"] == "omega":
					it_omega = it["max"]
			if not it_omega:
				it_omega = self.omega[i]
			it_nodes = self.n_nodes[i]
			it_score = self.get_best_ind(index=i)[0]

			# Check if optimization converge or reject it
			x, y_min, y_max, y_av = self._rearrange_pop(i, score)
			conv_av = np.mean(y_av[-y_av.size/12])
			if conv_av < 0.05:
				continue

			# If couple omega/ampli already existsn average with previous one
			for j, om in enumerate(omega):
				for k, no in enumerate(nodes):
					if om[0] == it_omega and no[0] == it_nodes and j == k:
						score[j].append(it_score)
						nodes[j].append(it_nodes)
						omega[j].append(it_omega)
						duplicate = True

			if duplicate == False:
				score.append([it_score])
				omega.append([it_omega])
				nodes.append([it_nodes])

		# Average points with multiple values
		n_av = []
		score_std = np.zeros(len(score))
		for i in range(len(score)):
			if  len(score[i]) > 1:
				n_av.append(len(score[i]))
			score_std[i] = np.std(np.array(score[i]))
			score[i] = sum(score[i]) / len(score[i])
			nodes[i] = sum(nodes[i]) / len(nodes[i])
			omega[i] = sum(omega[i]) / len(omega[i])

		if len(n_av) != 0:
			print(" -- Averaging " + str(len(n_av)) + " graph points with on average " + \
				num2str(float(sum(n_av)/len(n_av))) + " data sets for each --")

		# Sort lists
		nodes, omega, score, score_std = (list(t) for t in zip(*sorted(zip(nodes, omega, score, score_std))))
		freq = np.array(omega) / (2 * np.pi)


		x = []; y = [];	z = []; w = [];	j = 0;  n_prec = 0
		for i, n in enumerate(nodes):
			if n != n_prec:
				w.append(n)
				x.append([freq[i]])
				y.append([score[i]])
				z.append([score_std[i]])
				j += 1
			else:
				x[j-1].append(freq[i])
				y[j-1].append(score[i])
				z[j-1].append(score_std[i])
			n_prec = n
		fig, ax = Plot.initPlot()
		for i in range(0, len(x)):
		 	ax.errorbar(x[i], y[i], yerr=z[i], fmt='.-', \
		 		linewidth=1.5, label="$N_{nodes} = $ " + \
		 		num2str(w[i]))

		# Plot score as a fct of power in a loglog graph for different omega values
		# r_nodes = len(set(nodes))
		# n_nodes = len(nodes) / r_nodes
		# fig, ax = Plot.initPlot()
		# for i in range(r_nodes):
		# 	ax.errorbar(freq[n_nodes*i:n_nodes*i+n_nodes], np.array(score[n_nodes*i:n_nodes*i+n_nodes]), \
		# 		yerr=np.array(score_std[n_nodes*i:n_nodes*i+n_nodes]), fmt='.-', ecolor='r', \
		# 		linewidth=1.5)#, label="$N_{nodes} = $ " + \
		# 		#num2str(nodes[i*n_nodes]))
		# # Print 3DB bandwidth
		# major_ticks = np.arange(0, 10, 1)                                              
		# minor_ticks = np.arange(0, 10, 0.1)                                               
		# ax.set_xticks(major_ticks)                                                       
		# ax.set_xticks(minor_ticks, minor=True)  
		# ax.grid(which='both') 
		# ax.plot([min(freq), max(freq)], [max(score)/2, max(score)/2], linewidth=1.5, label="-3 dB score value")


		plt.title("Evolution of score for different structures in function of global frequency")
		Plot.configurePlot(fig, ax, 'Frequency','Score', legendLocation='lower right', size='small')
		ax.set_xlim([0, np.max(freq)])
		#ax.set_ylim([1e-3, 5])
		if show: plt.show()
		if save: plt.savefig(folder + filename + ".png", format='png', dpi=300)

	def pareto_dist(self, filename="results_pareto_dist", show=False, save=True):
		"""Perform specific analysis for a pareto dist batch"""
		
		folder = "pareto_pic/"
		mkdir_p(folder)

		d_ref = []
		power = []
		omega = []
		sim_time = self.sim_time[0]
		opt_type = self.opt_type[0]

		# Fill values from loaded variables
		for i in range(len(self.y)):

			duplicate = False
			assert self.sim_time[i] == sim_time, \
				"For a meaningfull graph, ensure the simulation times are the same for all data"
			assert self.opt_type[i] == opt_type, \
				"For a meaningfull graph, ensure the optimization algorithms are the same for all data"

			# Fetch iteration nodes number value and best individu
			it_d_ref = self.ref_dist[i]
			best = self.get_best_ind(index=i)
			it_omega = self.omega[best[1]]
			it_power = self.y_p[best[1]][best[2]]

			# If reference distance already exists, average with previous values
			for j, d in enumerate(d_ref):
				for k, om in enumerate(omega):
					#print d[0], om, it_d_ref, it_omega
					if d[0] == it_d_ref and om[0] == it_omega and j == k:
						power[j].append(it_power)
						d_ref[j].append(it_d_ref)
						omega[j].append(it_omega)
						duplicate = True

			if duplicate == False:
				power.append([it_power])
				d_ref.append([it_d_ref])
				omega.append([it_omega])

		# Average points with multiple values
		n_av = [0]
		power_inv = []
		power_std = []
		for i in range(len(power)):
			if  len(d_ref[i]) > 1:
				n_av.append(len(d_ref[i]))
			power_inv.append(list(map((lambda x: 1. / x), power[i])))
			power_std.append(np.std(np.array(power_inv[i])))
			power_inv[i] = sum(power_inv[i]) / len(power_inv[i])
			d_ref[i] = sum(d_ref[i]) / len(d_ref[i])
			omega[i] = sum(omega[i]) / len(omega[i])

		if len(n_av) != 0:
			print(" -- Averaging " + str(len(n_av)) + " graph points with on average " + \
				num2str(float(sum(n_av)/len(n_av))) + " data sets for each --")

		# Sort lists
		omega, d_ref, power_inv, power_std = (list(t) for t in (zip(*sorted(zip(omega, d_ref, power_inv, power_std)))))
		v_ref = list(map((lambda x: x / sim_time), d_ref))
		n_omega = len(omega) / len(set(omega))

		# Plot distance and power as a fonction of the nodes number
		fig, ax = Plot.initPlot()
		for i in range(len(set(omega))):
			ax.errorbar(v_ref[n_omega*i:n_omega*i+n_omega], power_inv[n_omega*i:n_omega*i+n_omega], \
				yerr=np.array(power_std[n_omega*i:n_omega*i+n_omega]), fmt='.-', \
				linewidth=1.5, label="$f = $ " + num2str(omega[n_omega*i]/2/np.pi) + " Hz")

		plt.title("Power evolution under constrained speed")
		plt.xlim([0, max(v_ref)])
		Plot.configurePlot(fig, ax, 'Speed','Power$^{-1}$', legendLocation='upper right', size='small')
		if show: plt.show()
		if save:
			print(" -- Print distance pareto in " + folder + filename + ".png --")
			plt.savefig(folder + filename + ".png", format='png', dpi=300)

	def pareto_power(self, filename="results_pareto_power", movieAnalysis=False, show=False, save=True):
		"""Perform specific analysis for a pareto power batch"""

		folder = "pareto_pic/"
		mkdir_p(folder)

		p_ref = []
		dist = []
		omega = []
		sim_time = self.sim_time[0]
		opt_type = self.opt_type[0]

		# Fill values from loaded variables
		for i in range(len(self.y)):

			duplicate = False
			assert self.sim_time[i] == sim_time, \
				"For a meaningfull graph, ensure the simulation times are the same for all data"
			assert self.opt_type[i] == opt_type, \
				"For a meaningfull graph, ensure the optimization algorithms are the same for all data"

			# Fetch iteration nodes number value and best individu
			it_p_ref = self.ref_pow[i]
			best = self.get_best_ind(index=i)
			it_omega = self.omega[best[1]]
			it_dist = self.y_d[best[1]][best[2]]

			# If reference distance already exists, average with previous values
			for j, p in enumerate(p_ref):
				for k, om in enumerate(omega):
					if p[0] == it_p_ref and om[0] == it_omega and j == k:
						p_ref[j].append(it_p_ref)
						dist[j].append(it_dist)
						omega[j].append(it_omega)
						duplicate = True

			if duplicate == False:
				dist.append([it_dist])
				p_ref.append([it_p_ref])
				omega.append([it_omega])

				# Limit cycle and video gait analysis
				if movieAnalysis:
					name  = str(int(it_omega/2/np.pi)) +"_" + str(int(it_p_ref))
					simName = folder + "sim_" + name
					pcaName = folder + "pca_" + name
					pcaTitle = "Limit cycle for $f = " + num2str(np.ceil(it_omega/2/np.pi)) + \
						" Hz$ and $P = " + num2str(np.ceil(it_p_ref)) + " W$"
				 	self.simulate_ind(best[1], best[2], simTime=10, movie=True, rc=False, pca=True, \
				 		simName=simName, pcaFilename=pcaName, pcaTitle=pcaTitle)


		# Average points with multiple values
		n_av = 0
		speed = []
		speed_std = []
		for i in range(len(dist)):
			n_av += len(dist[i])
			speed.append(list(map((lambda x: x / sim_time), dist[i])))
			speed_std.append(np.std(np.array(speed[i])))
			speed[i] = sum(speed[i]) / len(speed[i])
			p_ref[i] = sum(p_ref[i]) / len(p_ref[i])
			omega[i] = sum(omega[i]) / len(omega[i])


		if n_av != len(dist):
			print(" -- Averaging " + str(n_av) + " graph points with on average " + \
				num2str(float(n_av)/len(dist)) + " data sets for each --")

		# Sort lists
		omega, p_ref, speed, speed_std = (list(t) for t in zip(*sorted(zip(omega, p_ref, speed, speed_std))))
		n_omega = len(omega) / len(set(omega))

		# Plot distance and power as a fonction of the nodes number
		fig, ax = Plot.initPlot()
		for i in range(len(set(omega))):
			ax.errorbar(p_ref[n_omega*i:n_omega*i+n_omega], speed[n_omega*i:n_omega*i+n_omega], \
				yerr=speed_std[n_omega*i:n_omega*i+n_omega], fmt='.-', \
				linewidth=1.5, label="$f = $ " + num2str(omega[n_omega*i]/2/np.pi) + " Hz")

		plt.title("Speed evolution under constrained power")
		plt.xlim([0, max(p_ref)])
		Plot.configurePlot(fig, ax, 'Power','Speed', legendLocation='lower right', size='small')
		if show: plt.show()
		if save:
			print(" -- Print power pareto in " + folder + filename + ".png --")
			plt.savefig(folder + filename + ".png", format='png', dpi=300)

	def nodes_CL(self, filename="results_nodes_cl", noiseAnalysis=False, videoAnalysis=False, \
		genAnalysis=True, show=False, save=True):
		"""Perform specific analysis for closed loop experiments for different structures"""

		folder = "nodes_CL_pic/"
		mkdir_p(folder)

		nodes = []
		nrmse = []
		dist = []
		dist_cl = []
		sim_time = self.sim_time[0]
		opt_type = self.opt_type[0]
		sim_time_cl = 200
		max_it = 2

		# Fill values from loaded variables
		for i in range(len(self.y)):

			duplicate = False
			assert self.sim_time[i] == sim_time, \
				"For a meaningfull graph, ensure the simulation times are the same for all data"
			assert self.opt_type[i] == opt_type, \
				"For a meaningfull graph, ensure the optimization algorithms are the same for all data"

			# Fetch iteration nodes number value and best individu
			it_nodes = self.n_nodes[i]
			best = self.get_best_ind(index=i)
			it_dist = float(self.dists[best[1]][best[2]])

			# Find and eliminate duplicated experiments
			for j, no in enumerate(nodes):
				if no[0] == it_nodes:
					s, d, p, err = self.simulate_ind(best[1], best[2], simTime=sim_time_cl, movie=False, \
						openPhase=0.1, rc=True, nrmse=True, alpha=0.0001, beta=1, transPhase=0, trainingPhase=0.9)
					nodes[i].append(it_nodes)
					nrmse[i].append(err[0])
					dist_cl[i].append(d)
					dist[i].append(it_dist)
					duplicate = True	
			if duplicate == False:
					s, d, p, err = self.simulate_ind(best[1], best[2], simTime=sim_time_cl, movie=False, \
						openPhase=0.1, rc=True, nrmse=True, alpha=0.0001, beta=1, transPhase=0, trainingPhase=0.9)
					nodes.append([it_nodes])
					nrmse.append([err[0]])
					dist_cl.append([d])
					dist.append([it_dist])


		# Average points
		n_av = 0
		nrmse_std = []
		dist_cl_std = []
		dist_std = []
		for i in range(len(dist)):
			n_av += len(dist[i])
			nrmse_std.append(np.std(np.array(nrmse[i])))
			dist_cl_std.append(np.std(np.array(dist_cl[i]) / sim_time_cl * sim_time))
			dist_std.append(np.std(np.array(dist[i])))
			nrmse[i] = sum(nrmse[i]) / len(nrmse[i])
			dist_cl[i] = sum(dist_cl[i]) / len(dist_cl[i]) / sim_time_cl * sim_time
			dist[i] = sum(dist[i]) / len(dist[i])
			nodes[i] = sum(nodes[i]) / len(nodes[i])

		if n_av != len(dist):
			print(" -- Averaging " + str(n_av) + " graph points with on average " + \
				num2str(float(n_av)/len(nodes)) + " data sets for each --")

		# Sort lists
		nodes, nrmse, dist, dist_cl, nrmse_std, dist_std, dist_cl_std = \
			(list(t) for t in zip(*sorted(zip(nodes, nrmse, dist, dist_cl, nrmse_std, dist_std, dist_cl_std))))

		# Plot nrmse as a fonction of the nodes number
		fig, ax = Plot.initPlot()
		ax.errorbar(nodes, nrmse, yerr=nrmse_std, linewidth=1.5, fmt=".-", label="NRMSE")
		plt.title("NRMSE evolution for nodes number")
		Plot.configurePlot(fig, ax, 'Nodes','NRMSE', legendLocation='lower right', size='small')
		if show: plt.show()
		if save:
			print(" -- Print NRMSE in " + folder + filename + "_nrmse.png --")
			plt.savefig(folder + filename + "_nrmse.png", format='png', dpi=300)

		# Plot distance as a fonction of the nodes number
		fig, ax = Plot.initPlot()
		ax.errorbar(nodes, dist, yerr=dist__std, linewidth=1.5,  fmt=".-", \
			color=self._get_style_colors()[0], label="Open loop distance")
		ax.errorbar(nodes, dist_cl, yerr=dist_cl_std, linewidth=1.5, fmt=".-", \
			color=self._get_style_colors()[1], label="Closed-loop distance")
		plt.title("Travelled distance in function of nodes number")
		Plot.configurePlot(fig, ax, 'Nodes','Travelled distance', legendLocation='lower right',\
		 size='small')
		if show: plt.show()
		if save:
			print(" -- Print DISTS in " + folder + filename + "_dist.png --")
			plt.savefig(folder + filename + "_dist.png", format='png', dpi=300)