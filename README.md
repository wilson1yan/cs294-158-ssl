# Introduction

This is the repo for CS294-158 self-supervised learning demos.

# Setting Up

The conda environment can be created using `environment.yml`. If you run into issues, you can create your own empty environment on Python 3.7.6 with the following packages (feel free to use a different cuda version):
* conda install pytorch=1.4.0 torchvision=0.5.0 cudatooklkit=10.1 -c pytorch
* pip install requests
* pip install opencv-python
* pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git

# Training

You can execute the scripts in the `run` folder to train models on different self-supervised tasks. Note that different models may use a different number of GPUS (maximum 4).

Contrastive Predictive Coding (CPC) is still a work in progress.