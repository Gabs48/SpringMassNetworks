
= Spring Mass Networks =

== Description ==
This set of scripts provides custom simulations, optimization and learning tools to study the emergence of robust and efficient gaits for Mass Spring Damper Networks (MSDN) structures. As indicated in their name, they are constituted of massive shapeless nodes connected together by springs and dampers. The resting length of the spring can be actuated by a timed function so that the structures start moving. Random, genetic and CMA optimzation can be applied to optimize that function to provide the MSDN with an efficient gait. Different types of simulations allow to use a custom controller which can perform supervized learning to reproduce the expected outputs from its own state.

![alt tag](https://raw.githubusercontent.com/Gabs48/SpringMassNetworks/master/doc/msdn.img)

== Requirements ==
- Tested OS:  Windows, Ubuntu 14.04, scientific linux
- Python 2.7 on
- Required packages: numpy, matplotlib, neurolab, sqlite3, ffmpeg

== Usage Example ==

=== Training (finding best optimized individu through evolution) ===

1. Edit the desired simulation time and number of iteration variable in the code:
```
cd SpringMassNetworks
nano train.py
trainingIt = YOUR_DESIRED_NUMBER_OF_ITERATIONS
simTime = YOUR_DESIRED_SIMULTATION_TIME
```

2. Execute the training experiment with the desired type:
```
cd SpringMassNetworks
./train.py OPTIONS
```

3. OPTIONS are defined in the train script and are currently: *pareto, simtime, nodes, km, ref, powereff, noise*. More details on their goals can be found in the code itself.

4. Training is a long process and can take many hours, expecially with a high number of iterations and with numerous optimization batch. You can therefore split the processing into different cores on an optimization level using MPI:
```
mpirun -np NUMBER_OF_PROCESSORS ./train.py OPTIONS
```
5. The results

=== Analyzing (reproducing single simulation, with or withou learning and visual output and plot some analysis curves) ===

1. Fill the training folder result in the analysis script:
```
cd SpringMassNetworks
nano train.py

# Create Analysis object and load results folder
an = Analysis(root="YOUR_RESULTS_FOLDER", folder="THE_DESIRED_RESULT_FOLDER_IN_IT")
an.load()
```

2. In the same script, uncomment the lines corresponding to the analysis you want to perform. For example, to produce a video of the best individu simulation for 50 second with supervised learning in reservoir computing:
```
# Simulate best individu
score, index1, index2 = an.get_best_ind()
an.simulate_ind(index1, index2, simTime=50, movie=True, rc=True)
```

3. More information about the different available analysis are commented or can be found in the file *roboTraining/analysis.py*