#!/bin/sh

##########################
### General options
##########################

### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J LTS_Original

### -- ask for number of cores (default: 1) --
#BSUB -n 1

### -- Ask for 1 core machine 
#BSUB -R "span[hosts=1]"

### Ask for a GPU with 32GB of memory
#BSUB -R "select[gpu32gb]"

### Ask for NVLINK - Meaning: 

### -- Select the resources: 1 gpu in exclusive proce   ss mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

# request 20GB of system-memory pr core
#BSUB -R "rusage[mem=72000]"

### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s164440@student.dtu.dk

### -- send notification at completion--
#BSUB -N

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o ./output/_%J.out
#BSUB -e ./output/_%J.err
# -- end of LSF options --

module unload cuda
module unload cudann
module unload python

module load tensorflow/1.12-gpu-python-3.6.2
module load numpy/1.13.1-python-3.6.2-openblas-0.2.20
#module load tensorflow

nvidia-smi

/appl/cuda/9.2/samples/bin/x86_64/linux/release/deviceQuery



# Setup virtual env
#
export PYTHONPATH=
python3 -m venv mlenv
source mlenv/bin/activate



# Upgrade pip
pip3 install -U pip

# install 
pip install -r requirements.txt

# Expand path
#export PATH="$HOME/bin:$PATH"
#export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH

#
# Install basic python math
#
# Go to root

python3 train_Sony.py
