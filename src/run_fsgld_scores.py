#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import time
from importlib import import_module

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import load_dataset
from utils.metrics import agreement, total_variation_distance
from utils.tools_dl import predictions
from utils.toy_tools_data import fusion, pprint
from utils.uncertainties_tools import BS, ECE, NLL

print("os.path.abspath(__file__) =\n\t", os.path.abspath(__file__))
path = os.path.abspath('..')

# Print the local torch version
print(f"Torch Version {torch.__version__}")

# Save the user choice of settings
parser = argparse.ArgumentParser()
parser.add_argument('--method', default="fald", help='set the method')
parser.add_argument('--num_clients', default=20, type=int, help='set the number of clients')
parser.add_argument('-m', '--model', default="resnet20_cifar10", help='set the model name')
parser.add_argument('-n', '--num_iter', default=10, type=int, help='set the number of epochs')
parser.add_argument('-d', '--dataset_in', default="MNIST", choices=["MNIST", "CIFAR10", "CIFAR100"],
                    help='set the dataset')
parser.add_argument('--dataset_out', default="FashionMNIST")
parser.add_argument('-l', '--learning_rate', default=0.1, type=float, help='set the learning rate')
parser.add_argument('-N', '--batch_size', default=128, type=int, help='set the mini-batch size')
parser.add_argument('-p', '--num_local_steps', default=20, type=int, help='number of local steps')
parser.add_argument('-t', '--tau', default=1., type=float,
                    help='percentage of data coming from the training distribution')
parser.add_argument('-b', '--burnin', default=0, type=int, help='set the burn in period')
parser.add_argument('-g', '--ngpu', default=1, type=int, help='setl the number of gpus')
args = parser.parse_args()

# Define the title to save the results
title = str()
for key, value in {'m_': args.model, 'n_': args.num_iter, 'l_': args.learning_rate,
                   'N_': args.batch_size, 'num_local_steps': args.num_local_steps}.items():
    title += '-' + key + str(value)

# Save the path to store the data
path_workdir = './'
path_dataset = path_workdir + '../../dataset/'
path_figures = path_workdir + f'figures/'
path_variables = path_workdir + f'fsgld{title}/'
path_txt = path_variables + f"{args.dataset_out}.txt"
path_stats = path_variables + f'{args.dataset_out}{title}'

# Start the timer
startTime = time.time()

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
print(device)

# Define the network
model_dict = {'logistic': 'LogisticMnist', 'lenet5': 'LeNet5Mnist', 'resnet20': 'Resnet20Cifar10', 'mlp': 'MLPMnist'}
models = import_module('models.' + args.model)
model_cfg = getattr(models, model_dict[args.model])
transform = model_cfg.transform
net = model_cfg.base(*model_cfg.args, **model_cfg.kwargs)
net.to(device)

# Define the transformation
pprint('--- Preparing data ---')

# Load the dataset function
if args.dataset_out == 'CIFAR10':
    print('--- Load the dataset')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = getattr(load_dataset, "load_" + args.dataset_out)  # needed to compute agreement and total variation

    with open(path_dataset + '/cifar10_probs.csv', 'r') as fp:
        reference = np.loadtxt(fp)

else:
    dataset = getattr(torchvision.datasets, args.dataset_out)

# Define the parameter of the dataset
params_test = {"root": path_dataset, "transform": transform, "train": False, "download": True}

# Load the datasets
batch_size = 500
testset = dataset(**params_test)
testloader1 = DataLoader(testset, batch_size, shuffle=False)

# Load the In distribution dataset
dataset = getattr(torchvision.datasets, args.dataset_in)
testset = dataset(**params_test)
testloader0 = DataLoader(testset, batch_size, shuffle=False)

# Merge the datasets
testloader = fusion(testloader0, testloader1, args.tau)
ytest = testloader.dataset.tensors[1].numpy()

# For the statistics
save_dict = {"final_acc": [], "ece": [], "bs": [], "nll": []}
if args.dataset_out == 'CIFAR10':
    save_dict["agreement"] = []
    save_dict["total_variation_distance"] = []
tau_list = np.linspace(0, 1, num=100)

pprint("--- Compute the scores ---")

for pth in os.listdir(path_variables):

    path_samples = os.path.join(path_variables, pth)
    if not os.path.isdir(path_samples):
        continue

    all_probs = None

    for it, f in enumerate(os.listdir(path_samples)):

        if 'client' in f:
            continue
        if int(f) < args.burnin:
            print("Skip " + f)
            continue

        path = os.path.join(path_samples, f)

        # Compute the predictions
        probs = predictions(testloader, net, path).cpu().numpy()

        if all_probs is None:
            all_probs = probs
        else:
            all_probs = (it * all_probs + probs) / (it + 1)

        preds = np.argmax(probs, axis=1)

        print(f'Iter = {it + 1}, keep ' + f + f', accuracy = {np.round(100 * np.mean(preds == ytest), 1)}')

    # Compute the final accuracy
    preds_ = np.argmax(all_probs, axis=1)
    save_dict["final_acc"].append(100 * np.mean(preds_ == ytest))

    print('--- Final accuracy =', np.round(save_dict["final_acc"][-1], 1))

    # Load the HMC reference predictions
    if args.dataset_out == 'CIFAR10':
        # Now we can compute the metrics
        method_agreement = agreement(all_probs, reference)
        method_total_variation_distance = total_variation_distance(all_probs, reference)

        # Save the metrics
        save_dict["agreement"].append(method_agreement)
        save_dict["total_variation_distance"].append(method_total_variation_distance)

        # Print the scores
        print("Agreement =", method_agreement, "Total variation =", method_total_variation_distance)

    # Compute the Expected Calibration Error (ECE)
    save_dict["ece"].append(ECE(all_probs, ytest, num_bins=20))

    # Compute the Brier Score
    save_dict["bs"].append(BS(ytest, all_probs))

    # Compute the Negative Log Likelihood (NLL)
    save_dict["nll"].append(NLL(all_probs, ytest))

pprint('--- Save the results ---')

# Write the results
pprint(save_dict["final_acc"])
with open(path_txt, 'w') as f:
    f.write('\t---' + "fsgld".upper() + '---\n\n')
    for key, value in save_dict.items():
        f.write('%s: %s / %s\n' % (key, np.median(value), np.diff(np.percentile(value, [25, 75]))))

# Save the statistics
save_dict["ytest"] = ytest
save_dict["all_probs"] = all_probs
save_dict["tau_list"] = tau_list

# Save the dictionary
if os.path.exists(path_stats):
    saved_dict = torch.load(path_stats)
    save_dict.update(saved_dict)
torch.save(save_dict, path_stats)
