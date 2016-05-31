""" plot of comparison of different GA settings """


import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from roboTraining.utils import num2str, Plot



xmin = 0
xmax = 22

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
params = {'text.latex.preamble' : [r'\usepackage{mathtools}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)

def color(data, min , max):
	colors = ['blue'] * len(data)
	for i in range ( len(data)):
		if data[i] < min:
			colors[i] ='red'
		elif data[i] > max:
			colors[i] = 'green'
	return colors


def plot(title, names, data,std, min , max, xlabel = "", name = ""):
	fig, ax = plt.subplots(facecolor=(1,1,1))
	
	plt.subplots_adjust(top = 0.85)
	plt.subplots_adjust(bottom = 0.2)
	index = np.arange(len(names))
	barWidth = 0.4
	centerWidth = (1 -barWidth) / 2
	alpha = 1
	errorConfig = {'ecolor': '0.5', 'elinewidth' : 4, 'capthick' : 4,  'capsize' : 10}
	colors = color(data, min, max)
	rects1 = plt.bar(index + centerWidth, data, barWidth,
		color= colors,
		alpha = alpha,
		yerr=std,
		error_kw=errorConfig,
		)
	fontsize = 35
	#plt.title(title, fontsize = 28)
	plt.ylabel("Distance Traveled [m]", fontsize = fontsize)
	plt.xlabel(xlabel, fontsize = fontsize)
	ax.patch.set_facecolor((1,1,1))
	plt.ylim(xmin, xmax)

	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)

	plt.xticks(centerWidth + index + barWidth/2 , names)
	plt.tick_params(axis='x', which='both', bottom='on',top='off')
	plt.tick_params(axis='y', which='both', left='off',right='off')
	plt.tick_params(labelsize=fontsize)
	ax.yaxis.grid()
	for index, value in zip(index, data):
		plt.text(index + centerWidth + barWidth / 2, value + 2, str(value), 
			fontsize = fontsize, horizontalalignment = 'center', color = colors[index])
	plt.tight_layout()
	plt.show(block = False)
	Plot.save2eps(fig, name)
	plt.close()
	

def pmut(name):
	title = r"$p_{ \rm{mutation} }$"
	label = [ '0.01', '0.1', '0.33', '1']
	data = [8.97, 12.23, 13.23, 13.12]
	sigmaavg =  0.43
	min = 10
	max = 13
	plot("pmut", label, data, sigmaavg, min, max, title, name)

def sigmut(name):
	title = r"$ \sigma_{ \rm{mutation } }$"
	label = [ '0.01', '0.05', '0.1', '0.2', '0.5']
	data = [10.92, 10.73, 11.50, 12.09, 13.81]
	sigmaavg = 0.46
	min = 10
	max = 13
	plot("sigmut", label, data, sigmaavg, min, max, title, name)

def pcrossover(name):
	title = r"$ p_{\rm{crossover}}$ "
	label =  [ '0', '0.1', '0.33', '1']
	data = [11.24, 11.59, 12.25, 12.47]
	sigmaavg = 0.43
	min = 10
	max = 12
	plot("crossoverrate", label, data, sigmaavg, min, max, title, name)

def crossover(name):
	title = ""
	label = ['Uniform', 'Node', 'Type']
	data = [16.11, 17.35, 17.31]
	sigmaavg = 0.65
	min = 0
	max = 17
	plot("crossovertype", label, data, sigmaavg, min, max, title, name)

def scaling(name):
	title = ""
	label = ['Expon-\nential', 'Linear', 'Rank', 'Sigma \n Truncation']
	data = [18.14, 16.54, 15.63, 17.39]
	sigmaavg = 0.75
	min = 16
	max = 17
	plot("scaling", label, data, sigmaavg, min, max, title, name)

def selector(name):
	title = ""
	label =  [ 'tournament', 'roulette']
	data = [17.5, 16.3]
	sigmaavg = 0.53
	min = 0
	max = 17
	plot("selector", label, data, sigmaavg, min, max, title, name)

def population(name):
	title = "Population Variables \n (Number of Generations - PopSize)"
	label = ['20 - 75', '30 - 50', '50 - 30', '75 -20']
	data = [17.92, 18.07, 18.35, 16.42]
	sigmaavg = 1.011 
	min = 0
	max = 17.5
	plot("popgen", label, data, sigmaavg, min, max, title, name)

pmut("GAMETApmut")
sigmut("GAMETAsmut")
pcrossover("GAMETApcross")
crossover("GAMETAtcross")
scaling("GAMETAscale")
selector("GAMETAsel")
population("GAMETApop")