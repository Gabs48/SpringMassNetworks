#!/bin/bash

# Options
#PBS -N LongPowerEffJob
#PBS -o stdout.$PBS_JOBID
#PBS -e stderr.$PBS_JOBID
#PBS -l walltime=60:00:00
#PBS -l nodes=3:ppn=1
#PBS -m e

# Environment
cd $PBS_O_WORKDIR
echo "== Start HPC Job =="
date
n_proc=$( cat $PBS_NODEFILE  |  wc  -l )
echo "MPI Nodes number= ${n_proc}"
module load Python/2.7.11-intel-2016a
module load matplotlib/1.5.1-intel-2016a-Python-2.7.11-freetype-2.6.3
module load freetype/2.6.3-intel-2016a
module load mpi4py

# Start script
mpirun -np $n_proc ./example_hpc.py simtime

# Finalize
echo "== End HPC Job =="
