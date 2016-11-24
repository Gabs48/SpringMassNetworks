#! /usr/bin/env python2

import os
from roboTraining.analysis import *

if __name__ == "__main__":

	# Create Analysis object and load results folder
	an = Analysis(root="paper", folder="cl")
	an.load()

	# General plots
	# for u in ["score", "distance", "power"]:
	# 	an.plot_all_raws(unit=u)
	# 	an.plot_all_raws_av(unit=u)
	#	an.plot_all_gens()
	# an.plot_all_conv_errs()
	# an.plot_all_state_spaces()
	# an.plot_all_params()

	# Simulate best individu
	#score, index1, index2 = an.get_best_ind()
	#print index1, index2
	# an.simulate_ind(index1, index2, simTime=1, movie=True, simNoise=0, rc=False)

	# Simtime analysis
	# an.simtime()

	# Pareto Anaysis
	# an.pareto()
	#an.pareto_dist()
	# an.pareto_power()

	# Spring Mass values analysis
	# an.km()

	# Omega value analysis
	#an.freq()

	# Nodes number value analysis
	# an.nodes()
	an.nodes_CL()

	# Noise analysis
	# an.plot_all_noise_sims()
	# an.plot_all_noise_params()