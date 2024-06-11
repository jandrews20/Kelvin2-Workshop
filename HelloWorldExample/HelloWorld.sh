#!/bin/bash

#SBATCH --job-name=HelloWorld
#SBATCH --mem 4000M
#SBATCH --time=00:02:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=HelloWorldOutput.out



module load apps/python3/3.10.5/gcc-9.3.0

python3 main.py 8
