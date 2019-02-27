#!/bin/sh

# Start with python 2

# Setup virtual env
pip install virtualenv

export PYTHONPATH=
virtualenv mlpy2env
source mlpy2env/bin/activate

# Upgrade pip
pip install -U pip

# install 
pip install -r requirements.txt

# stop virtual env
deactivate
chmod -R 777 mlpy2env

module unload python
module load python3/3.6.2

#
# Setup virtual env
#
export PYTHONPATH=
python -m venv mlpy3env
source mlpy3env/bin/activate

# Upgrade pip
pip install -U pip

# install 
pip install -r requirements.txt

# stop virtual env
deactivate
chmod -R 777 mlpy3env

