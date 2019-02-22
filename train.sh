#!/bin/sh

##########################
### General options
##########################

### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J LearnToSeeInTheDark

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- Ask for 4 core machine 
#BSUB -R "span[hosts=1]"

### Ask for a GPU with 32GB of memory
#BSUB -R "select[gpu32gb]"

### Ask for NVLINK - Meaning: 
### BSUB -R "select[sxm2]"

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=2:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

# request 20GB of system-memory pr core
#BSUB -R "rusage[mem=20000]"

### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s164440@student.dtu.dk

### -- send notification at start --
#BSUB -B

### -- send notification at completion--
#BSUB -N

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o _%J.out
#BSUB -e _%J.err
# -- end of LSF options --

module unload cuda
module unload cudann
module unload python

module load numpy/1.13.1-python-3.6.2-openblas-0.2.20

nvidia-smi
# Load the cuda module
module load cuda/9.2
module load cudnn/v7.4.2.24-prod-cuda-9.2

/appl/cuda/9.2/samples/bin/x86_64/linux/release/deviceQuery


# Setup virtual env
#
export PYTHONPATH=
python3 -m venv stdpy3
source stdpy3/bin/activate

#
# Upgrade pip
#
pip3 install -U pip


# Expand path
export PATH="$HOME/bin:$PATH"
export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH

#
# Install basic python math
#
pip3 install -U numpy
pip3 install -U scipy
pip3 install -U torch
pip3 install -U pillow
pip3 install -U rawpy

python3 train.py
