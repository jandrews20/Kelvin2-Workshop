#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --job-name MNISTClassification
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output=Output_MNIST.out

module add apps/python3/3.10.5/gcc-9.3.0
module add libs/nvidia-cuda/11.7.0/bin
module add apps/pip_python310/22.3.1/python3-3.10.5

# CUDA 11.8
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib
pip install scikit-learn

User = whoami

export PYTHONPATH=$PYTHONPATH:/users/$User/gridware/share/python/3.10.5/lib/python3.10/site-packages/

srun --ntasks=1 python3 main.py &

wait
