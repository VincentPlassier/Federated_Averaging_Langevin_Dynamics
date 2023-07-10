#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import time
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch._C import default_generator
from torch.utils.data import DataLoader

from algo_dl.sgld import Sgld
from utils.tools_dl import accuracy_model
from utils.toy_tools_data import pprint

# Save the user choice of settings 
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_iter', default=5, type=int, help='set the number of epochs')
parser.add_argument('-d', '--dataset_name', default="CIFAR10", choices=["MNIST", "CIFAR10", "CIFAR100", "SVHN"],
                    help='set the dataset')
parser.add_argument('-m', '--model', default="resnet", help='set the model name')
parser.add_argument('-l', '--learning_rate', default=1e-6, type=float, help='set the learning rate')
parser.add_argument('-N', '--batch_size', default=16, type=int, help='set the mini-batch size')
parser.add_argument('-t', '--thinning', default=1, type=int, help='set the thinning')
parser.add_argument('-b', '--burnin', default=0, type=int, help='set the burn in period')
parser.add_argument('-w', '--weight_decay', default=5, type=float, help='set the l2 regularization parameter')
parser.add_argument('-p', '--precondition_decay_rate', default=1, type=float,
                    help='set the decay rate of rescaling of the preconditioner')
parser.add_argument('--save_samples', default=True, action='store_true', help='if True we save the samples')
parser.add_argument('-g', '--ngpu', default=1, type=int, help='setl the number of gpus')
parser.add_argument('-s', '--seed', default=-1, type=int, help='set the seed')
parser.add_argument("--pretrained_model", default=False, action='store_true', help="Use pretrained model")
args = parser.parse_args()

# Define the title to save the results
title = str()
for key, value in {'m_':                       args.model, 'n_': args.num_iter, 'l_': args.learning_rate,
                   'N_':                       args.batch_size,
                   'precondition_decay_rate_': args.precondition_decay_rate}.items():
    title += '-' + key + str(value)

# Print the local torch version
print(f"\nTorch Version {torch.__version__}")
print("\nos.path.abspath(__file__) =\n\t%s\n" % os.path.abspath(__file__))

# Save the path to store the data
path_workdir = './'
path_dataset = path_workdir + '../../dataset/'
path_figures = path_workdir + f'figures/'
path_variables = path_workdir + f'sgld{title}/'
path_samples = path_variables + f"samples-{args.seed}/"

# Create the directory if it does not exist
os.makedirs(path_dataset, exist_ok=True)
os.makedirs(path_figures, exist_ok=True)
os.makedirs(path_variables, exist_ok=True)
if args.save_samples:
    os.makedirs(path_samples, exist_ok=True)

# Set random seed for reproducibility
seed_np = args.seed if args.seed != -1 else None
seed_torch = args.seed if args.seed != -1 else default_generator.seed()
np.random.seed(seed_np)
torch.manual_seed(seed_torch)
torch.cuda.manual_seed(seed_torch)

# Define the network
model_dict = {'logistic': 'LogisticMnist', 'lenet5': 'LeNet5Mnist', 'resnet': 'Resnet20Cifar10', 'mlp': 'MLPMnist'}
models = import_module('models.' + args.model)
model_cfg = getattr(models, model_dict[args.model])
if args.model == 'resnet':
    model_cfg.kwargs = {'pretrained': args.pretrained_model}
net = model_cfg.base(*model_cfg.args, **model_cfg.kwargs)
transform = model_cfg.transform

# Define the transformation
pprint('--- Preparing data ---')

# Load the function associated with the chosen dataset
dataset = getattr(torchvision.datasets, args.dataset_name)

# Define the parameter of the dataset
params_train = {"root": path_dataset, "transform": transform, "train": True, "download": True}
params_test = {"root": path_dataset, "transform": transform, "train": False}

# Load the datasets
batch_size_test = 500
trainset = dataset(**params_train)
testset = dataset(**params_test)
trainloader = DataLoader(trainset, args.batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size_test, shuffle=False)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
print(device)

# Handle multi-gpu if desired
net.to(device)
if (device.type == 'cuda') and (args.ngpu > 1):
    net = nn.DataParallel(net, list(range(args.ngpu)))

# Store the scores
save_dict = vars(args)
save_dict['Accuracy Beginning:'] = accuracy_model(net, testloader, device, verbose=False)
print('Accuracy Beginning:', save_dict['Accuracy Beginning:'])

# Define the parameters of the SGLD optimizer
speudo_batches = len(trainset)  # / args.batch_size  # we multiply the gradient by this quantity
num_burn_in_steps = args.burnin * (speudo_batches // args.batch_size + (speudo_batches % args.batch_size > 0))
params_optimizer = {"lr":                      args.learning_rate, "num_pseudo_batches": speudo_batches,
                    "precondition_decay_rate": args.precondition_decay_rate, "num_burn_in_steps": num_burn_in_steps}

# Define the Sgld sampler
model = Sgld(net)

# Start the timer
startTime = time.time()

# Run the SGLD algorithm
model_state_dict, save_stats = model.run(trainloader, testloader, args.num_iter, args.weight_decay / speudo_batches,
                                         params_optimizer, args.burnin, args.thinning, 0, args.save_samples,
                                         path_samples)

# End the timer
executionTime = time.time() - startTime
save_dict["execution time"] = executionTime
print("Execution time =", executionTime)
