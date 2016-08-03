#! /usr/bin/env python2

from roboTraining.analysis import *

if __name__ == "__main__":

	# Create Analysis object and load results folder
	an = Analysis(folder="simtime_5000_it_GA_fail/SimTime")#pareto_CMA_25s_5000it/")#test_uniformite_25_s")#convergence_test/data/")#
	an.load()

	# General plots
	# an.plot_all_scores()
	# an.plot_all_gens()
	# an.plot_all_conv_errs()
	# an.plot_all_scores_av()
	# an.plot_all_state_spaces()

	# Simulate best individu
	# score, index1, index2 = an.get_best_ind()
	# an.simulate_ind(50, index1, index2)

	# Simtime analysis
	an.simtime()

	# Pareto Anaysis
	# an.pareto()

	# Spring Mass values analysis
	# an.km()

	# Nodes number value analysis
	# an.nodes()