source mlpy2env/bin/activate

# install 
pip install -r requirements.txt

deactivate

module unload python
module load numpy/1.13.1-python-3.6.2-openblas-0.2.20

source mlpy3env/bin/activate

# install 
pip install -r requirements.txt

deactivate