# RLab Common
Common methods and tools for data scrubbing, vectorized forecasting and ETL work. I can no longer privately maintain these pieces of work, so now it is public. 

## How to Unit Test
```
sudo apt install $(cat work/apt-packages.txt)
cd rlab-common/work
bash venv-init.sh
source venv/bin/activate
pip install -r requirements.txt
bash reinstall-project.sh
less logs/*.out
```
Some files may fail because of indentation.

## How to Explore
```
sudo apt install $(cat work/apt-packages.txt)
cd rlab-common
bash venv-init.sh
source venv/bin/activate
pip install -r work/requirements.txt
pip install work/wheel/*.whl
jupyter-notebook --ip=127.0.0.1
```

