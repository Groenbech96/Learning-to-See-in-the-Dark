#!/bin/sh

# Start with python 2

# Upgrade pip
pip install -U pip

# Setup virtual env
pip install virtualenv

export PYTHONPATH=
virtualenv mlpy2env
source mlpy2env/bin/activate

# install 
pip install -r requirements.txt

module unload python
module load numpy/1.13.1-python-3.6.2-openblas-0.2.20

# stop virtual env
deactivate

chmod -R 777 mlpy2env

#
# Setup virtual env
#
export PYTHONPATH=
python3 -m venv mlpy3env
source mlpy3env/bin/activate

# Upgrade pip
pip install -U pip

# install 
pip install -r requirements.txt

# stop virtual env
deactivate
chmod -R 777 mlpy3env

