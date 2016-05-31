""" plot comparison between different training methods """

from roboTraining.utils import Plot
import matplotlib.pyplot as plt
import numpy as np
from roboTraining.utils import num2str, CTE_POWERPOINT

def allRunsPlot():
	data = np.genfromtxt('Experiments/Final/robotComparison.csv', names = True, delimiter=';')
	noTypes = len(data)

	fig, ax = plt.subplots(facecolor=(1,1,1), figsize=(10, 6), dpi=150)

	index = np.arange(noTypes)
	bar_width = 0.3
	alpha = 1
	error_config = errorConfig = {'ecolor': '0.5', 'elinewidth' : 4, 'capthick' : 4,  'capsize' : 10}
	label = []
	for i in range(noTypes):
		label.append( num2str(data["noNodes"][i]) +"\n"+ num2str(data["noNeighbours"][i]))
		#label.append( num2str(data["noNodes"][i]) +"\n nodes \n" + num2str(data["noNeighbours"][i]) + "\nNeigh.")

	rects1 = plt.bar(index, data['gaAvg'], bar_width,
		color='#5555ff',
		alpha = alpha,
		yerr=data['gaStd']/np.sqrt(10),
		error_kw=error_config,
		label='GA')

	rects2 = plt.bar(index + bar_width, data['randAvg'], bar_width,
		color='#ff5555',
		alpha = alpha,
		yerr=data['randStd']/np.sqrt(10),
		error_kw=error_config,
		label='Random')

	rects3 = plt.bar(index + 2 *  bar_width, data['cmaAvg'], bar_width,
		color='#555555',
		alpha = alpha,
		yerr=data['cmaStd']/np.sqrt(10),
		error_kw=error_config,
		label='CMA-ES')

	if CTE_POWERPOINT:
		fontsize = 22
		markersize = 30
		mew = 5
	else:
		fontsize = 14
		markersize = 10
		mew = 5

	plt.xlabel('Number of Nodes \n Number of Connections per Node', fontsize = fontsize)
	plt.ylabel('Distance Traveled [m]', fontsize = fontsize)
	plt.xticks(index + bar_width, label)


	plt.plot(index + 0.15, data["gaMin"],"k_", markersize = markersize, mew = mew)
	plt.plot(index + 0.15, data["gaMax"],"k_", markersize = markersize, mew = mew)

	plt.plot(index + 0.45, data["randMin"],"k_", markersize = markersize, mew = mew)
	plt.plot(index + 0.45, data["randMax"],"k_", markersize = markersize, mew = mew)

	plt.plot(index + 0.75, data["cmaMin"],"k_", markersize = markersize, mew = mew)
	plt.plot(index + 0.75, data["cmaMax"],"k_", markersize = markersize, mew = mew)


	plt.legend(prop={'size':fontsize})
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.yaxis.grid() # vertical lines
	#ax.spines['bottom'].set_visible(False)
	#ax.spines['left'].set_visible(False)
	plt.tick_params(labelsize=fontsize)
	plt.tick_params(axis='x', which='both', bottom='on',top='off')
	plt.tick_params(axis='y', which='both', left='off',right='off')
	plt.tight_layout()
	Plot.save2eps(fig, "comparison")
	plt.show()

def specificRunPlot():
	"""config uit GA Training run0 en CMATraining run0:	
	20 nodes 3 neighbours voor 10 sec laten evalueren"""
	data = np.genfromtxt('Experiments/Final/cmavsga.csv', delimiter=';')
	resultsCMA = data[0,:]
	resultsGA = data[1,:]
	x = np.arange(len(resultsCMA))
	fig, ax = Plot.initPlot()
	ax.plot(x, resultsCMA, 'b.', label = 'CMA-ES')
	ax.plot(x, resultsGA, 'r.', label = 'GA')
	Plot.configurePlot(fig, ax, 'iteration', 'Distance Travelled', legend = True)
	plt.show(block = False)
	Plot.save2eps(fig, 'GAvsCMA')
	

#specificRunPlot()
allRunsPlot()