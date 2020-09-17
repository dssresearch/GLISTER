#GLISTER, a GeneraLIzation based data Subset selecTion for Efficient and Robust learning framework

This repo contains implementation of both GLISTER-ONLINE and GLISTER-ACTIVE under various diffrent 

# Dependencies

To run this code fully, you'll need [PyTorch](https://pytorch.org/) (we're using version 1.4.0) and [scikit-learn](https://scikit-learn.org/stable/).
We've been running our code in Python 3.7.

# Running an experiment

`python3 run_dss.py <data directory> <data name>`\

`python run.py --model resnet --nQuery 1000 --data CIFAR10 --alg badge`\
runs an active learning experiment using a ResNet and CIFAR-10 data, querying batches of 1,000 samples according to the BADGE algorithm.
This code allows you to also run each of the baseline algorithms used in our paper. 

`python run.py --model mlp --nQuery 10000 --did 6 --alg conf`\
runs an active learning experiment using an MLP and dataset number 6 from OpenML, querying batches of 10,000 with confidence sampling.
Note that in our code, OpenML datasets can only be used with MLP architectures.
 
# Analyzing experimental results
See the readme file in `scripts/` for more details about generating plots like those in our paper.

