source mlpy2env/bin/activate

# install 
pip install -r requirements.txt

deactivate
chmod -R 777 mlpy2env

module unload python
module load python3/3.6.2

source mlpy3env/bin/activate

# install 
pip install -r requirements.txt

deactivate
chmod -R 777 mlpy3env