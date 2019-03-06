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
#BSUB -R "rusage[mem=64000]"

### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s164440@student.dtu.dk

### -- send notification at completion--
#BSUB -N

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o ./output/tensorflow/train/_%J.out
#BSUB -e ./output/tensorflow/train/_%J.err
# -- end of LSF options --

module unload cuda
module unload cudann

module load cuda/9.0
module load cudnn/v7.4.2.24-prod-cuda-9.0

nvidia-smi

/appl/cuda/9.2/samples/bin/x86_64/linux/release/deviceQuery

# Setup virtual env
pip install virtualenv

export PYTHONPATH=
source mlpy2env/bin/activate

# Upgrade pip
pip install -U pip

# install 
pip install -r requirements.txt

python train_sony.py
