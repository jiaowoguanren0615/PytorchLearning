<h1 align='center'>Checklist & Python Package</h1>

## 1. Miniconda(recommend) or Anaconda

## ([Miniconda](https://docs.conda.io/projects/miniconda/en/latest/))

## ([Anaconda](https://www.anaconda.com/download/))


## 2. Create and activate a conda virtual environment 
```shell
conda create -n torch2 python=3.9

conda activate torch2
```

## 3. Install PyTorch package
### Recommend to install the GPU version
[pytorch-website](https://pytorch.org/)

## 4. Install Tensorboard package
```shell
pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
```



## 5. Install Graphviz and torchviz package 
###  Install Graphviz 
([Download | Graphviz](https://www.graphviz.org/download/))

### 	Install torchviz
```shell
pip install torchviz -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 6. Install other packages
```shell
pip install -U jupyter numpy pandas matplotlib seaborn tqdm mglearn torchinfo transformers scikit-learn nltk Pillow timm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 7. Add more conda environments to jupyter
##  (step1) check your conda env package by running: "conda list" or "pip list" after activating your virtual env. Run step1 command if you don't have the package named ipykernel. If you had installed this package, then run step2 command.
```shell
conda install ipykernel
```

##  (step2) python -m ipykernel install --user --name YourEnvName --display-name "Name in Jupyter" 
### For example: I create a conda-env named "torch2", I want it to show the name "T2" in jupyter. Then run this follow code:
```shell
python -m ipykernel install --user --name torch2 --display-name "T2"
```

## 8. Edit jupyter-lab(notebook) configuration
### For jupyter-lab
```shell
jupyter lab --generate-config

vim ~/.jupyter/jupyter_lab_config.py
```
### For jupyter-notebook
```shell
jupyter notebook --generate-config

vim ~/.jupyter/jupyter_notebook_config.py
```
