#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --job-name MNISTClassification
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output=Output_MNIST.out

module add apps/python3/3.10.5/gcc-9.3.0
module add libs/nvidia-cuda/11.7.0/bin

export PYTHONPATH=$PYTHONPATH:/users/40237845/gridware/share/python/3.10.5/lib/python3.10/site-packages/

srun --ntasks=1 python3 main.py &

wait