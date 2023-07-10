import argparse
import os
import time
import warnings

import proplot as pplt
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from utils.toy_tools_data import *
from utils.toy_tools_func import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_name', default='breast-cancer', help='set the dataset')
parser.add_argument('--method', default=True, action='store_true',
                    help='list of the methods')
parser.add_argument('-n', '--num_clients', default=5, type=int, help='set the number of clients')
parser.add_argument('-m', '--mc_iter', default=200, type=int, help='set the number of Monte Carlo iterations')
parser.add_argument('-b', '--batch_size', default=1, type=int, help='set the mini-batch size')
parser.add_argument('-t', '--temperature', default=1, type=float, help='set the l2 regularization parameter')
parser.add_argument('-l', '--l2', default=1., type=float, help='set the l2 regularization parameter')
parser.add_argument('-s', '--seed', default=-1, type=int, help='set the seed')
args = parser.parse_args()

# Define the title to save the results
title = args.dataset_name
for key, value in {'method_': args.method, 'numclients_': args.num_clients, 'm_': args.mc_iter,
                   'batch_':  args.batch_size, 'seed_': args.seed}.items():
    title += '-' + key + str(value)

# Save the path to store the data
path_workdir = './'
path_dataset = path_workdir + '../../dataset'
path_figures = path_workdir + 'figures/'
path_variables = path_workdir + 'variables/'
path_stats = path_variables + title
path_txt = path_variables + 'text-' + title + '.txt'

# Create the directory if it does not exist
os.makedirs(path_figures, exist_ok=True)
os.makedirs(path_variables, exist_ok=True)

# set random seed for reproducibility
seed = args.seed if args.seed != -1 else None
rng = np.random.default_rng(1)
np.random.seed(seed)

pprint('--- Get Dataset ---')

if args.dataset_name == 'covertype':
    inputs, targets = fetch_covtype(data_home=path_dataset, download_if_missing=True, return_X_y=True, random_state=42)

    # Transform to binary dataset
    idx = np.where(targets <= 2)[0]
    inputs = inputs[idx]
    targets = targets[idx]

    inputs = StandardScaler().fit_transform(inputs)
    targets = LabelEncoder().fit_transform(targets)

    # Subsample
    max_data_size = np.inf
    if len(targets) > max_data_size:
        id_subset = rng.choice(len(targets), max_data_size, replace=False)
        inputs = inputs[id_subset]
        targets = targets[id_subset]
        print(f'inputs = {np.shape(inputs)}, targets = {np.shape(targets)}')

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(inputs, targets, test_size=0.2, random_state=42)

else:
    Xtrain, Ytrain, Xtest, Ytest = load_UCI_dataset(args.dataset_name, rng, prop=.8)

pprint('--- Generate the Loss functions ---')

input_dim = Xtrain.shape[1]
output_dim = len(np.unique(Ytrain))
if output_dim == 2:  # for binary logistic regression
    output_dim = 1

oe = OneHotEncoder(sparse=False).fit(Ytrain.reshape(-1, 1))
transform = oe.transform if output_dim > 2 else None

if output_dim > 2:
    Ytrain = oe.transform(Ytrain.reshape(-1, 1))
    Ytest = oe.transform(Ytest.reshape(-1, 1))

# Logistic regression
Logistic = LogisticGradSto if output_dim > 2 else BinaryLogisticGradSto

# define the function to test the performance
grad_train = Logistic(Xtrain, Ytrain, args.batch_size, args.temperature, args.l2, oe)
grad_test = Logistic(Xtest, Ytest, args.batch_size, args.temperature, args.l2, oe)

pprint('--- Start the calculation procedure ---')

# We obtain the best parameter in function of the learning rate: lr

if args.method:
    method = ['ula', 'fald', 'vr_fald_star']
elif isinstance(args.method, list) and args.method in ['ula', 'fald',, 'vr_fald_star']:
    method = args.method
else:
    raise NameError('args.method should be a list.')

methods_list = [*method]


def load(path_variables, args_key, name_to_load):
    keys = args_key + [name_to_load]

    for f in os.listdir(path_variables):
        if not f.endswith(".npy"):
            continue
        if np.fromiter(map(lambda s: s not in f, keys), dtype=float).sum() == 1 and f[
                                                                                    :len(name_to_load)] == name_to_load:
            return np.load(os.path.join(path_variables, f), allow_pickle=True)


# Define arguments to find the sample
args_key = title.split('-')

# Load the saved samples
saved_params = dict.fromkeys(methods_list)
for method in methods_list:
    saved_params[method] = load(path_variables, args_key, method)

# Start the timer
startTime = time.time()

predictive_dict = {}
loss_dict = {}

for method in methods_list:
    it = 0
    if method == 'ula':
        burnin = int(.1 * len(saved_params))
        thinning = 1
    else:
        burnin = int(.1 * len(saved_params))
        thinning = 1
    sub_samples = saved_params[method][burnin::thinning]
    print(f'--- Number of saved parameters = {len(sub_samples)} ---\n')
    loss_dict[method] = {'train': [], 'test': []}
    predictive_dict[method] = {'train': [], 'test': []}
    for it, param in tqdm(enumerate(sub_samples)):
        loss_dict[method]['train'].append(grad_train.loss(param))
        loss_dict[method]['test'].append(grad_test.loss(param))
        if output_dim == 1:
            predictive_dict[method]['train'] = average(it, predictive_dict[method]['train'],
                                                       1 / (1 + np.exp(- np.dot(Xtrain, param))))
            predictive_dict[method]['test'] = average(it, predictive_dict[method]['test'],
                                                      1 / (1 + np.exp(- np.dot(Xtest, param))))
        else:
            predictive_dict[method]['train'] = average(it, predictive_dict[method]['train'],
                                                       softmax(- np.dot(Xtrain, param), axis=1))
            predictive_dict[method]['test'] = average(it, predictive_dict[method]['test'],
                                                      softmax(- np.dot(Xtest, param), axis=1))

np.save(os.path.join(path_variables, 'predictive_dict-' + title), predictive_dict)
np.save(os.path.join(path_variables, 'loss_dict-' + title), loss_dict)

# End the timer
executionTime = time.time() - startTime
print("Execution time =", executionTime)

pprint('--- Display the results ---')

# Display the results
fig, axs = pplt.subplots(share=0, ncols=2, refwidth=2)
for name, error in loss_dict.items():
    for it, train in enumerate(['train', 'test']):
        axs[it].plot(error[train], label=name + '_' + train)
axs.set_yscale('log')
axs.format(grid=True, xlabel=r'Iteration', ylabel=r"Loss", yformatter='log', fontsize=12)
fig.legend(title='Loss comparison', ncols=5, loc='bottom', prop={'size': 14}, order='C', frameon=True, shadow=True,
           edgecolor='k', facecolor='gray2')
fig.savefig(os.path.join(path_figures, title + '.pdf'), bbox_inches='tight')

# Store the score
save_dict = vars(args)
save_dict["execution time"] = executionTime
