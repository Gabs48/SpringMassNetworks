#! /usr/bin/env python2

import os
from roboTraining.analysis import *

if __name__ == "__main__":

	# Create Analysis object and load results folder
	an = Analysis(folder="powereff_low_E_d_with_k/data/Machine-0")
	an.load()

	# Find best individu
	score, index1, index2 = an.get_best_ind()

	# Try different learning phases and close-loop phases
	for trans in [0.05]:#[0, 0.1, 0.2, 0.3]:
		for train in [0.4]:#[0.4, 0.55, 0.7]:
			for ol in [0.1]:#[0, 0.1, 0.2]:
				for alpha in [0.1]:#0.001, 0.1, 1, 10, 100, 1000]:#, 10000]:
					for beta in [0.9]:#, 0.6]:
						an.simulate_ind(index1, index2, simTime=100, movie=False, rc=True, transPhase=trans, \
							trainingPhase=train, openPhase=ol, alpha=alpha, beta=beta)