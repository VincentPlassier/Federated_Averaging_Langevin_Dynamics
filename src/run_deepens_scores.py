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
from utils.toy_tools_data import pprint
from utils.uncertainties_tools import accuracy_confidence, BS, calibration_curve, confidence, ECE, NLL

print("os.path.abspath(__file__) =\n\t", os.path.abspath(__file__))
path = os.path.abspath('..')

# Print the local torch version
print(f"Torch Version {torch.__version__}")

# Save the user choice of settings 
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default="resnet", help='set the model name')
parser.add_argument('-n', '--num_epochs', default=10, type=int, help='set the number of epochs')
parser.add_argument('-d', '--dataset_in', default="MNIST", choices=["MNIST", "CIFAR10", "CIFAR100"],
                    help='set the dataset')
parser.add_argument('--dataset_out', default="FashionMNIST")
parser.add_argument('-l', '--learning_rate', default=0.1, type=float, help='set the learning rate')
parser.add_argument('-N', '--batch_size', default=128, type=int, help='set the mini-batch size')
parser.add_argument('-t', '--tau', default=1., type=float,
                    help='percentage of data coming from the training distribution')
parser.add_argument('-g', '--ngpu', default=1, type=int, help='setl the number of gpus')
args = parser.parse_args()

title = str()
for key, value in {'m_': args.model, 'n_': args.num_epochs, 'l_': np.round(args.learning_rate, 3),
                   'N_': args.batch_size, 't_': args.tau}.items():
    title += '-' + key + str(value)

# Save the path to store the data
path_workdir = './'
path_dataset = path_workdir + '../../dataset/'
path_figures = path_workdir + f'figures/'
path_variables = path_workdir + f'deep_ensembles{title.split("-t_")[0]}/'
path_samples = path_variables + f"samples/"
path_txt = path_variables + f"{args.dataset_out}.txt"
path_stats = path_variables + f'{args.dataset_out}{title.split("-t_")[1]}'

# Start the timer
startTime = time.time()

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
print(device)

# Define the network
model_dict = {'logistic': 'LogisticMnist', 'lenet5': 'LeNet5Mnist', 'resnet': 'Resnet20Cifar10', 'mlp': 'MLPMnist'}
models = import_module('models.' + args.model)
model_cfg = getattr(models, model_dict[args.model])
transform_in = model_cfg.transform
net = model_cfg.base(*model_cfg.args, **model_cfg.kwargs)
net.to(device)

# Define the transformation
pprint('--- Preparing data ---')

# Load the dataset function
additional_scores = True
if args.dataset_out == 'CIFAR10' and additional_scores:
    print('--- Load the dataset of the challenge ---')
    transform_out = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = getattr(load_dataset, "load_" + args.dataset_out)  # needed to compute agreement and total variation

    with open(path_dataset + '/cifar10_probs.csv', 'r') as fp:
        reference = np.loadtxt(fp)

else:
    transform_out = transform_in
    dataset = getattr(torchvision.datasets, args.dataset_out)

# Define the parameters of the dataset
params_test_in = {"root": path_dataset, "transform": transform_in, "train": False, "download": True}
params_test_out = {"root": path_dataset, "transform": transform_out, "train": False}

# Load the Out distribution datasets
batch_size = 500
testset = dataset(**params_test_out)
testloader1 = DataLoader(testset, batch_size, shuffle=False)

# Load the In distribution dataset
dataset = getattr(torchvision.datasets, args.dataset_in)
testset = dataset(**params_test_in)
testloader0 = DataLoader(testset, batch_size, shuffle=False)

# Merge the datasets
# testloader = fusion(testloader0, testloader1, args.tau)  # from utils.toy_tools_data import fusion
# ytest = testloader.dataset.tensors[1].numpy()
testloader = testloader1

# Load the targets
ytest = np.loadtxt(path_dataset + '/cifar10_test_y.csv').astype(
    int) if args.dataset_out == 'CIFAR10' else testset.targets.numpy()

# For the statistics
save_dict = {"final_acc": [], "ece": [], "bs": [], "nll": []}
if args.dataset_out == 'CIFAR10' and additional_scores:
    save_dict["agreement"] = []
    save_dict["total_variation_distance"] = []
tau_list = np.linspace(0, 1, num=100)

pprint("--- Compute the scores ---")

for it, f in enumerate(os.listdir(path_samples)):

    it10 = it % 10

    path = os.path.join(path_samples, f)
    print(f'Sample path: {f}')

    if it10 == 0:
        print(f'\nInitialize, it = {it}, it10 = {it10}')
        all_probs = None

    # Compute the predictions
    probs = predictions(testloader, net, path).cpu().numpy()

    if all_probs is None:
        all_probs = probs
    else:
        all_probs = (it10 * all_probs + probs) / (it10 + 1)

    preds = np.argmax(probs, axis=1)

    print(f'Iter = {it + 1}, keep ' + f + f', accuracy = {np.round(100 * np.mean(preds == ytest), 1)}')

    if it10 != 9:
        continue

    # Compute the predictions
    all_probs = predictions(testloader, net, path).cpu().numpy()
    preds = np.argmax(all_probs, axis=1)
    save_dict["final_acc"].append(100 * np.mean(preds == ytest))

    print('--- Final accuracy =', np.round(save_dict["final_acc"][-1], 1))

    # Load the HMC reference predictions
    if args.dataset_out == 'CIFAR10' and additional_scores:
        # Now we can compute the metrics
        method_agreement = agreement(all_probs, reference)
        method_total_variation_distance = total_variation_distance(all_probs, reference)

        # Save the metrics
        save_dict["agreement"].append(method_agreement)
        save_dict["total_variation_distance"].append(method_total_variation_distance)

        # Print the scores
        print("Agreement =", method_agreement, "Total variation =", method_total_variation_distance)

    # Compute the accuracy in function of p(y|x)>tau
    accuracies, misclassified = confidence(ytest, all_probs, tau_list)

    # Compute the Expected Calibration Error (ECE)
    save_dict["ece"].append(ECE(all_probs, ytest, num_bins=20))

    # Compute the Brier Score
    save_dict["bs"].append(BS(ytest, all_probs))

    # Compute the accuracy - confidence
    acc_conf = accuracy_confidence(all_probs, ytest, tau_list, num_bins=20)

    # Compute the calibration curve
    cal_curve = calibration_curve(all_probs, ytest, num_bins=20)

    # Compute the Negative Log Likelihood (NLL)
    save_dict["nll"].append(NLL(all_probs, ytest))

pprint(f'--- Save the results ---\n{path_txt}')

# Write the results
with open(path_txt, 'w') as f:
    f.write('\t---' + "Deep Ensemble".upper() + '---\n\n')
    for key, value in save_dict.items():
        f.write('%s: %s / %s\n' % (key, np.mean(value), np.std(value)))

# Save the statistics
save_dict["ytest"] = ytest
save_dict["all_probs"] = all_probs
save_dict["tau_list"] = tau_list
save_dict["accuracies"] = accuracies
save_dict["calibration_curve"] = cal_curve
save_dict["accuracy_confidence"] = acc_conf

# Save the dictionary
if os.path.exists(path_stats):
    saved_dict = torch.load(path_stats)
    save_dict.update(saved_dict)
torch.save(save_dict, path_stats)
