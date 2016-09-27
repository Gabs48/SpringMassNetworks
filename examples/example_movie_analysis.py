#! /usr/bin/env python2

import os
from roboTraining.analysis import *

if __name__ == "__main__":

	# Simulate best individu in each folder
	i = 0
	for path, subdirs, files in os.walk("/home/gabs48/edu/Data/powereff_low_E_d_with_k/data"):
		if not subdirs:
			if not os.path.isfile(path + "/sim.mp4"):
				an = Analysis(root=path, folder=".")
				an.load()
				score, index1, index2 = an.get_best_ind()
				an.simulate_ind(index1, index2, simName=path + "/sim" , movie=False, rc=False)
			else:
				i += 1
				#print "Deja fait: " + str(i)