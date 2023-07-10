# !/usr/bin/env python
# coding: utf-8

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from utils.toy_tools_data import create_title, pprint

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mc_iter', default=10000, type=int, help='set the number of Monte Carlo iterations')
parser.add_argument('-t', '--tau', default=1., type=float)
parser.add_argument('-p', '--prob_update_param', default=1, type=float, help='set the probability of communication')
parser.add_argument('-f', '--factor', default=1, type=float)
args = parser.parse_args()

# Define the number of workers
num_clients = 5

# Define the number of iterations
mc_iter = args.mc_iter

# Set the privacy parameter
tau = args.tau

# Probability of communication
pc = args.prob_update_param

# Define the specific title
title = create_title(mc_iter, num_clients, factor=args.factor, pc=pc, tau=tau)

# Save the path to store the data
path_workdir = './'
path_figures = path_workdir + f'figures/'
path_variables = path_workdir + f'{title}/'

# Define the used methods
method_keys = {'fald', 'vr_fald_star', 'qlsd', 'qlsdpp', 'dglmc'}
legend_name = {'fald': 'FALD', 'vr_fald_star': r'Vr-FALD$^\star$', 'qlsd': 'QLSD', 'qlsdpp': r'QLSD$++$',
               'dglmc': 'DG-LMC'}

# Define the MSE dictionary
mse_dict = dict.fromkeys(method_keys)

# Define the necessary arguments
args_key = [*title.split("-"), 'mse_dict.npy']

pprint('--- Start the calculations ---')

# Start the timer
startTime = time.time()

for f in tqdm(os.listdir(path_variables)):
    if not f.endswith(".npy"):
        continue
    if False not in list(map(lambda s: s in f, args_key)):
        mse = np.load(os.path.join(path_variables, f), allow_pickle=True).ravel()[0]
    else:
        continue
    for key, item in mse.items():
        if mse_dict[key] is None:
            mse_dict[key] = item
        else:
            mse_dict[key] = np.concatenate((mse_dict[key], item))

# Set the palette
sns.set_palette("Set1")  # colorblind, pastel, Set3

for key, item in mse_dict.items():
    if item is None:
        continue
    num_pts = 1000
    item = item[:, :len(item[1]) // 2]
    id_sub = np.unique(np.rint(10 ** (np.log(len(item[0])) / np.log(10) * np.linspace(0, 1, num_pts)) - 1)).astype(int)
    item = np.take(item, id_sub, axis=1)
    confidence = np.percentile(item, [5, 95], axis=0)
    plt.plot(id_sub, mean, ls='--', marker='o', ms=10, markevery=(.3, .25), markeredgecolor='k', markeredgewidth=1,
             label=legend_name[key])
    plt.fill_between(id_sub, confidence[0], confidence[1], alpha=0.2, linewidth=.2, edgecolor='k')

# plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Number of iteration", size=15)
plt.ylabel(r"Mean Square Error (MSE)", size=15)
plt.legend(fontsize=9.9,
           # loc='best',
           bbox_to_anchor=(1.05, -.18),
           ncol=5,
           # title="Algorithm",
           title_fontsize=12,
           frameon=True,
           shadow=True,
           facecolor='white',
           edgecolor='k',
           labelspacing=1,
           handlelength=2)

plt.savefig(os.path.join(path_figures, f'{title}-mse.pdf'), bbox_inches='tight')

# End the timer
executionTime = time.time() - startTime
print("Execution time =", executionTime)

# Write the MSE results
with open(path_variables + 'mse.txt', 'w') as f:
    f.write(f'executionTime = {executionTime}\n')
    for key, value in mse_dict.items():
        if value is None:
            continue
        burnin = - int(value.shape[1] / 50)
        f.write(f'\t--- {key.upper()} ---\n')
        f.write('%.2E +/- %.2E\n\n' % (value[:, burnin:].mean(), value[:, burnin:].std()))
