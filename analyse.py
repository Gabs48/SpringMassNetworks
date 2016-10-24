#! /usr/bin/env python2

import os
from roboTraining.analysis import *

if __name__ == "__main__":

	# Create Analysis object and load results folder
	an = Analysis(root=".", folder="t")
	an.load()

	# General plots
	for u in ["score", "distance", "power"]:
		an.plot_all_raws(unit=u)
		an.plot_all_raws_av(unit=u)
		an.plot_all_gens(unit=u)
	an.plot_all_conv_errs()
	an.plot_all_state_spaces()

	# Simulate best individu
	# score, index1, index2 = an.get_best_ind()
	# an.simulate_ind(index1, index2, simTime=10, movie=True, rc=False)

	# Simtime analysis
	# an.simtime()

	# Pareto Anaysis
	# an.pareto()
	an.pareto_dist()
	# an.pareto_power()

	# Spring Mass values analysis
	# an.km()

	# Omega value analysis
	# an.freq()

	# Nodes number value analysis
	# an.nodes()

	# Noise analysis
	# an.plot_all_noise_sims()
	# an.plot_all_noise_params()