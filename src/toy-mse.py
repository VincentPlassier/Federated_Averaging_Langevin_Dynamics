# !/usr/bin/env python
# coding: utf-8

import argparse
import os

from scipy.stats import multivariate_normal

from algo_np.fald_np import Fald
from algo_np.fald_star_np import FaldStar
from utils.toy_tools_data import *
from utils.toy_tools_func import *

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exp_num', default=None)
parser.add_argument('-n', '--niter', default=1, type=int)
parser.add_argument('-m', '--mc_iter', default=10000, type=int, help='set the number of Monte Carlo iterations')
parser.add_argument('-t', '--tau', default=1., type=float)
parser.add_argument('-p', '--prob_update_param', default=1., type=float, help='set the probability of communication')
parser.add_argument('-f', '--factor', default=1., type=float)
args = parser.parse_args()

# Define the number of iterations
niter, mc_iter, burn_in = args.niter, args.mc_iter, 0

# Define the number of workers
num_clients = 5

# Set the privacy parameter
tau = args.tau

# Probability of communication
pc = args.prob_update_param

# Define the specific title
title = create_title(mc_iter, num_clients, factor=args.factor, pc=pc, tau=tau, exp_num=args.exp_num)

pprint(title)

# Save the path to store the data
path_workdir = './'
path_figures = path_workdir + f'figures/'
path_variables = path_workdir + f'{title.split("-expNum_")[0]}/'

# Create the directory if it1 does not exist
os.makedirs(path_figures, exist_ok=True)
os.makedirs(path_variables, exist_ok=True)

# set random seed for reproducibility
seed = 3
rng = np.random.default_rng(seed)

pprint("--- Generate the dataset ---")

# define the number of samples own by each worker
num_samples = 100

# define the dimension
dim = 200

# define the centers of the clusters
means = 10 * rng.normal(size=(num_clients, dim))
cov_scales = ss.uniform(loc=.1, scale=10).rvs(size=num_clients, random_state=rng)
cov = [create_cov(dim, cs, psi=2 * np.pi, rng=rng) for cs in cov_scales]

cov_true = np.linalg.inv(np.sum([np.linalg.inv(sigma) for sigma in cov], axis=0))
mean_true = cov_true.dot(np.sum([np.dot(np.linalg.inv(sigma), mu) for (mu, sigma) in zip(means, cov)], axis=0))

# generate the dataset
X = [ss.multivariate_normal.rvs(mean=means[i], cov=sigma, size=num_samples, random_state=rng) for i, sigma in
     enumerate(cov)]
X_mean = np.mean(X, axis=1)
X_flatten = np.concatenate(X)
X_flatten_mean = X_flatten.mean(axis=0)

# display the dataset
n_components = 2
pca = PCA(n_components).fit(X_flatten)
for i, x in enumerate(X):
    x_pca = pca.transform(x)
    mu_pca = pca.transform(means[i].reshape(1, -1)).flatten()
    plt.scatter(x_pca[:, 0], x_pca[:, 1], marker=".", label=i)
    # plt.plot(mu_pca[0], mu_pca[1], '*', color='b', markersize=6)
plt.xlabel(r"First PCA coordinate", size=15)
plt.ylabel(r"Second PCA coordinate", size=15)
plt.savefig(os.path.join(path_figures, f'clients_fig-expNum_{args.exp_num}.pdf'), bbox_inches='tight')

# contain the potentials of each worker
gradU = []
for (mu, sigma) in zip(means, cov):
    gradU.append(GaussianToy(mu, np.linalg.inv(sigma)))

# set the learning rate of the clients
cov_eig = np.linalg.eig(cov_true)[0]
# define the time step size
gamma = 2 / (1 / min(cov_eig) + 1 / max(cov_eig))
print(gamma)
#
param_clients = np.tile(mean_true + 0 * np.ones(*mean_true.shape), (num_clients, 1))


def f1(x): return np.linalg.norm(x)


def f2(x): return np.linalg.norm(x - mean_true) ** 2


# The experiments are carried out on the next function
func = f2

pprint("--- Exact sampler ---")

num_samples_ref = 10 ** 1

# define the sampler algorithm
sampler = multivariate_normal(mean=mean_true, cov=cov_true, seed=rng)

# sampling steps
theta_ref = sampler.rvs(num_samples_ref)

# get the mala estimate
param_ref = np.fromiter(map(func, theta_ref), dtype=float).mean(axis=0)

# in the special case where func = ftest, we know exactly the true value of E[f(theta)]
print(f'np.linalg.eig(cov_true)[0].sum() - param_ref = {np.linalg.eig(cov_true)[0].sum() - param_ref}')
param_ref = np.trace(cov_true)

pprint("--- Compute the MSE ---")

# Probability to update the control variates
q = pc

# Define the quantization
s = 1
memory_coef = 1 / (1 + np.minimum(dim / s ** 2, np.sqrt(dim) / s))
quantization = StochasticQuantization(s).quantize

factor_gamma = args.factor * pc * num_clients

args_s = gradU, np.copy(param_clients), tau, pc, factor_gamma * gamma
args_vrs = gradU, np.copy(param_clients), tau, pc, q, factor_gamma * gamma
args_qlsd = gradU, quantization, factor_gamma * gamma / num_clients, np.copy(param_clients[0])
args_qlsdpp = gradU, quantization, factor_gamma * gamma / num_clients, np.copy(param_clients[0]), memory_coef
args_dglmc = gradU, pc, factor_gamma * gamma / num_clients, factor_gamma * gamma / num_clients, np.copy(param_clients)

print(f'{factor_gamma * gamma / num_clients}; 1 / max(cov_eig) = {1 / max(cov_eig)}')

fald_dict = {'fald': Fald, 'vr_fald_star': FaldStar, 'qlsd': Qlsd, 'qlsdpp': QlsdPp, 'dglmc': Dglmc}
mse_dict = {}

# Define the path to save the results
path_save = os.path.join(path_variables, title)

for key, item in fald_dict.items():

    if key == 'fald':
        args_key = args_s
    elif key == 'vr_fald_star':
        args_key = args_vrs
    elif key == 'qlsd':
        args_key = args_qlsd
    elif key == 'qlsdpp':
        args_key = args_qlsdpp
    elif key == 'dglmc':
        args_key = args_dglmc

    mse_dict[key] = mse_calculation(func, item, args_key, niter, mc_iter, burn_in, param_ref, True)
    # path_save + '-' + key)

    if mc_iter > 1e5:
        np.save(path_save + '-mse_dict', mse_dict)  # save at each iteration, not necessary

print(path_save[:-1], os.path.join(path_figures, title))
plot_save_fig(mse_dict, mc_iter, burn_in, os.path.join(path_figures, title), xlog=False)
np.save(path_save + '-mse_dict', mse_dict)
