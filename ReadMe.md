#                            Checklist & Python Package

## 1. Miniconda(recommend) or Anaconda

##     ([Miniconda — miniconda documentation](https://docs.conda.io/projects/miniconda/en/latest/))

##     ([Free Download | Anaconda](https://www.anaconda.com/download/))



## 2. Create and activate a conda virtual environment 

##     (Run this command: conda create -n torch2 python=3.9)

```shell
conda create -n torch2 python=3.9

conda activate torch2
```



## 3. Install PyTorch package ([(pytorch.org)](https://pytorch.org/))

## 	(Recommend to install the GPU version)



## 4. Install Tensorboard package

```shell
pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
```



## 5. Install Graphviz and torchviz package 

##     Install Graphviz ([Download | Graphviz](https://www.graphviz.org/download/))

## 	Install torchviz (pip install torchviz -i https://pypi.tuna.tsinghua.edu.cn/simple)

```shell
pip install torchviz -i https://pypi.tuna.tsinghua.edu.cn/simple
```



## 6. Install other packages

```shell
pip install -U jupyter numpy pandas matplotlib seaborn tqdm mglearn torchinfo transformers scikit-learn nltk gensim Pillow timm -i https://pypi.tuna.tsinghua.edu.cn/simple
```



## 7. Add conda env to jupyter

##     (step1) check your conda env package by running: "conda list" or "pip list" after activating your virtual env. Run step1 command if you don't have the package named ipykernel. If you had installed this package, then run step2 command.

```shell
conda install ipykernel
```



##     (step2) python -m ipykernel install --user --name YourEnvName --display-name "jupyter中显示名称"  比如我上面创建的环境名称为torch2，我想让它在jupyter中显示名称为T2，则命令是：

```shell
python -m ipykernel install --user --name torch2 --display-name "T2"
```

