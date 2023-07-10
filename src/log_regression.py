import argparse
import os
import time

from sklearn.datasets import fetch_covtype
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from algo_np.fald_np import Fald
from algo_np.vr_fald_star_np import VrFaldStar
from utils.toy_tools_data import *
from utils.toy_tools_func import *

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_name', default='covertype', help='set the dataset')
parser.add_argument('-n', '--num_clients', default=10, type=int, help='set the number of clients')
parser.add_argument('-m', '--mc_iter', default=200, type=int, help='set the number of Monte Carlo iterations')
parser.add_argument('-b', '--batch_size', default=1, type=int, help='set the mini-batch size')
parser.add_argument('-g', '--lr', default=5e-6, type=float, help='set the learning rate')
parser.add_argument('--prop_ula', default=1, type=float, help='set the proportion of data used for each SGLD update')
parser.add_argument('-t', '--temperature', default=1, type=float, help='set the l2 regularization parameter')
parser.add_argument('-l', '--l2', default=1., type=float, help='set the l2 regularization parameter')
parser.add_argument('-s', '--seed', default=-1, type=int, help='set the seed')
args = parser.parse_args()

# Define the title to save the results
title = args.dataset_name
for key, value in {'numclients_': args.num_clients, 'm_': args.mc_iter, 'g_': args.lr, 'ltwo_': args.l2,
                   'batch_':      args.batch_size, 'seed_': args.seed}.items():
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

# Set random seed for reproducibility
seed = args.seed if args.seed != -1 else None
rng = np.random.default_rng(1)
np.random.seed(seed)

pprint('--- Get Dataset ---')

#################################################################################################################
# regression_datasets = ['boston', 'concrete', 'naval', 'yacht', 'protein', 'winewhite']
# classification_datasets = ['wine', 'breast-cancer', 'titanic', 'mushroom', 'adult', 'covertype']
#################################################################################################################

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

#################################################################################################################
################################################ Start Algorithm ################################################
#################################################################################################################

pprint('--- Generate the heterogeneous dataset ---')

input_dim = Xtrain.shape[1]
output_dim = len(np.unique(Ytrain))
if output_dim == 2:  # for binary logistic regression
    output_dim = 1

oe = OneHotEncoder(sparse=False).fit(Ytrain.reshape(-1, 1))

transform = oe.transform if output_dim > 2 else None
datasets = generate_heterogeneous_classification(Xtrain, Ytrain, args.num_clients, transform, rng)

if output_dim > 2:
    Ytrain = oe.transform(Ytrain.reshape(-1, 1))
    Ytest = oe.transform(Ytest.reshape(-1, 1))

# Define the learning rate
if output_dim == 1:
    eigenvalues = np.linalg.eigvalsh(Xtrain.T.dot(Xtrain))
    lr = 2 / (eigenvalues[0] + eigenvalues[-1])
else:
    lr = args.lr
ratio = 1
lr = ratio * args.num_clients * lr
lr_dict = {'ula':  lr / args.num_clients, 'fald': lr, 'vr_fald_star': lr,
           'qlsd': lr / args.num_clients, 'qlsdpp': lr / args.num_clients, 'dglmc': lr / args.num_clients}

# Define the numbers of iterations
mc_iter_dict = {'ula':          args.mc_iter,
                'fald':         args.mc_iter,
                'vr_fald_star': args.mc_iter,
                'qlsd':         args.mc_iter,
                'qlsdpp':       args.mc_iter,
                'dglmc':        args.mc_iter, }

# Define the algorithms
fald_dict = {'fald': Fald, 'vr_fald_star': VrFaldStar, 'qlsd': Qlsd, 'qlsdpp': QlsdPp, 'dglmc': Dglmc}

# Define some thinning to subsample
thinning = 1 / lr if 1 / lr < 1000 else 1000

# probability of communication
pc = 1 / args.num_clients

# probability to update the control variates
q = 1 / args.num_clients

# set the privacy parameter
tau = 1

# Define the quantization
s = 1
memory_coef = 1 / (1 + np.minimum(input_dim / s ** 2, np.sqrt(input_dim) / s))
quantization = StochasticQuantization(s).quantize

# Define the batch_size
batch_size = int(pc * len(Ytrain) / args.num_clients) if args.batch_size == 0 else args.batch_size

# Define the logistic regression
Logistic = LogisticGradSto if output_dim > 2 else BinaryLogisticGradSto

# define the function to test the performance
grad_test = Logistic(Xtest, Ytest, batch_size, args.temperature, args.l2, oe)

# contain the potentials of each client
gradU = list()
for (Xi, Yi) in datasets:
    gradU.append(Logistic(Xi, Yi, batch_size, args.temperature, args.l2, oe))

# Just to compare
clf = LogisticRegression(fit_intercept=False, random_state=42, max_iter=1000).fit(Xtrain, Ytrain)

print(f'Accuracy = {(10 ** 2 * clf.score(Xtest, Ytest)).round(1)}%')

# Compute the predictive probabilities
# clf.predict_proba(Xtest)

# Give the coefficient
param_best = np.squeeze(clf.coef_)

# Compute the initial loss
loss_init = grad_test.loss(param_best)
print(f'Loss = {loss_init}')

param_clients = np.tile(np.copy(param_best), (args.num_clients, *[1] * param_best.ndim))
loss_init = grad_test.loss(np.copy(param_clients.mean(axis=0)))

niter = 1  # multiprocessing.cpu_count()
print(f'niter = {niter}')

# Start the timer
startTime = time.time()

# OLD TO DISAPPEAR
alpha = .1
error_dict = dict.fromkeys(fald_dict.keys(), np.zeros(args.mc_iter + 1))

#################################################################################################################
################################################# FALD sampler #################################################
#################################################################################################################

arguments = None

for it in range(niter):
    loss = np.zeros(args.mc_iter + 1)
    loss[0] = np.copy(loss_init)
    for name_meth, method in fald_dict.items():
        if name_meth == 'fald':
            arguments = gradU, np.copy(param_clients), tau, pc, lr_dict[name_meth]
        elif name_meth == 'vr_fald_star':
            arguments = gradU, np.copy(param_clients), tau, pc, q, lr_dict[name_meth]
        elif name_meth == 'qlsd':
            arguments = gradU, quantization, lr_dict[name_meth], np.copy(param_clients[0])
        elif name_meth == 'qlsdpp':
            arguments = gradU, quantization, lr_dict[name_meth], np.copy(param_clients[0]), memory_coef
        elif name_meth == 'dglmc':
            arguments = gradU, pc, lr_dict[name_meth], lr_dict[name_meth], np.copy(param_clients)
        algorithm = method(*arguments)
        # run the algorithm
        for i in tqdm(range(args.mc_iter)):
            update = algorithm.step()
            loss[i + 1] = grad_test.loss(algorithm.get_server_param()) if update else loss[i]
        error_dict[name_meth] = (np.copy(loss) + it * error_dict[name_meth]) / (it + 1)
        idx = np.linspace(0, len(algorithm.get_saved_params()) - 1, num=int(args.mc_iter / thinning), dtype=int)
        np.save(os.path.join(path_variables, name_meth + '-' + title),
                np.take(algorithm.get_saved_params(), idx, axis=0))

#################################################################################################################
################################################## ULA sampler ##################################################
#################################################################################################################

batch_size_ula = int(args.prop_ula * len(Ytrain))
logistic_train = Logistic(Xtrain, Ytrain, batch_size_ula, args.temperature, args.l2, oe)

ula = ULA(logistic_train.grad, lr_dict['ula'])
param_ula = np.copy(param_best)  # np.ones((input_dim, output_dim))
loss = np.zeros(mc_iter_dict['ula'] + 1)
loss[0] = np.copy(loss_init)
for i in tqdm(range(mc_iter_dict['ula'])):
    param_ula = ula.step(np.copy(param_ula))
    loss[i + 1] = grad_test.loss(np.copy(param_ula)) if i % int(1 / pc) == 0 else loss[i]
error_dict['ula'] = loss

# Save the subsamples
idx = np.linspace(0, len(ula.get_saved_params()) - 1, num=int(args.mc_iter / thinning), dtype=int)
np.save(os.path.join(path_variables, 'ula-' + title), np.take(ula.get_saved_params(), idx, axis=0))

# End the timer
executionTime = time.time() - startTime
print("Execution time =", executionTime)

plt.figure(2)
plt.xscale('log')
plt.yscale('log')
plt.grid('True')
for name, error in error_dict.items():
    plt.plot(error, label=name)
plt.axhline(y=loss_init, color='grey', linestyle='--', label='SGD')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss comparison')
plt.legend()
plt.savefig(os.path.join(path_figures, title + '.pdf'), bbox_inches='tight')

np.save(os.path.join(path_variables, 'error_dict-' + title), error_dict)

# Store the score
save_dict = vars(args)
save_dict["execution time"] = executionTime
save_dict["errors"] = [np.mean(error[int(.9 * args.mc_iter):]) for name, error in error_dict.items()]
