#! /usr/bin/env python2

import os
from roboTraining.analysis import *

if __name__ == "__main__":

	# Create Analysis object and load results folder
	an = Analysis(root=".", folder="Nodes")
	an.load()

	# General plots
	# an.plot_all_scores()
	# an.plot_all_gens()
	# an.plot_all_conv_errs()
	# an.plot_all_scores_av()
	# an.plot_all_state_spaces()

	# Simulate best individu
	# score, index1, index2 = an.get_best_ind()
	# an.simulate_ind(index1, index2, simTime=50, movie=True, rc=False)

	# Simtime analysis
	# an.simtime()

	# Pareto Anaysis
	# an.pareto()

	# Spring Mass values analysis
	# an.km()

	# Omega value analysis
	# an.freq()

	# Nodes number value analysis
	an.nodes()

	# Noise analysis
	# an.plot_all_noise_sims()
	# an.plot_all_noise_params()