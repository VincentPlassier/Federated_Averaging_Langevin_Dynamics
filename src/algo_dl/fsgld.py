#!/usr/bin/env python
# coding: utf-8

import copy
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm
from utils.sgld_tools import SGLD
from utils.tools_dl import accuracy_model


def client_preprocessing(loader, model, num_epochs = 10, weight_decay = 5, params_optimizer = None,
                         params_scheduler = None, it_freq = 1, it_burnin = 0, device = "cuda:0"):
    num_data = len(loader)
    model_state_dict = model.state_dict()
    mean = dict.fromkeys(model_state_dict)
    cov = dict.fromkeys(model_state_dict)
    model.train()
    # Define the optimizer
    criterion = nn.CrossEntropyLoss()
    params_optimizer["num_pseudo_batches"] = len(loader)  # we multiply the gradient by this quantity
    optimizer = SGLD(model.parameters(), **params_optimizer)
    params_scheduler["step_size_down"] = num_data
    params_scheduler["cycle_momentum"] = False
    scheduler = CyclicLR(optimizer, **params_scheduler)
    # Optimizer parameters
    # grad_clip = .1
    num_iter = 0
    # Start the optimization stage
    for epoch in range(num_epochs):
        running_loss = 0.
        total, correct = 0, 0
        for i, (inputs, labels) in enumerate(loader):
            # zero the parameter gradients
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            for param in model.parameters():
                loss += weight_decay / num_data * torch.norm(param) ** 2
            loss.backward()
            # nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            # update the learning rate
            scheduler.step()
            # update the running loss
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        with torch.no_grad():
            if epoch >= it_burnin and (epoch - it_burnin) % it_freq == 0:
                # Updates the Gaussian parameters
                for name, param in model_state_dict.items():
                    # Update the mean
                    if mean[name] is None:
                        mean[name] = param
                    else:
                        if not param.requires_grad:
                            continue
                        mean[name] += (num_iter * mean[name] + param) / (
                                num_iter + 1)
                    # Update the covariance
                    if cov[name] is None:
                        cov[name] = param ** 2
                    else:
                        cov[name] = (num_iter * cov[name] + (mean[name] - param) ** 2) / (
                                num_iter + 1)
                num_iter += 1
        print('Epoch %d, Accuracy = %.2f%%, Loss = %.3f\n' % (epoch + 1, 100 * correct / total, running_loss))
    # Define the precision matrix
    with torch.no_grad():
        precision = dict.fromkeys(model_state_dict)
        for name, param in cov.items():
            precision[name] = 1 / param
    return mean, cov  # precision


def init_net_clients(trainloader, model, param_preprocessing, load_init = True, path_fn = None, device = "cuda:0"):
    # list of the means and precisions on each worker
    means, precisions = [], []
    for i, loader_i in tqdm(enumerate(trainloader)):
        path_client = os.path.join(path_fn, f'client_{i}') if path_fn is not None else None
        if load_init and os.path.exists(path_client):
            # load the mean and precision of the client
            load_dict = torch.load(path_client)
            mean_i = load_dict["mean"]
            precision_i = load_dict["precision"]
        else:
            net_client = copy.deepcopy(model).to(device)
            # perform some preprocessing steps
            mean_i, precision_i = client_preprocessing(loader_i, net_client, **param_preprocessing)
            # store the model of the client
            load_dict_client = {'mean': mean_i, 'precision': precision_i}
            if path_client is not None:
                torch.save(load_dict_client, path_client)
        # store the neural networks
        means.append(mean_i)
        precisions.append(precision_i)
        # Compute SWAG statistics
        net = copy.deepcopy(model)
        net.load_state_dict(mean_i)
        net.to(device)
        acc_train = accuracy_model(net, loader_i, device)
        print('Train accuracy =', acc_train)
    return means, precisions


class Fsgld:

    def __init__(self, trainloader, testloader, net, params_preprocessing, load_init = True, path_fn = None):
        self.trainloader = trainloader
        self.testloader = testloader
        self.net = net
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_clients = len(trainloader)
        self.criterion = nn.CrossEntropyLoss()
        self.means, self.precisions = init_net_clients(trainloader, copy.deepcopy(net), params_preprocessing, load_init,
                                                       path_fn)
        # Store the statistics
        self.save_dict = {"epochs": [], "losses_test": [], "accuracies_test": [], "mse_relative": []}

    def net_update(self, num_local_steps, lr, weight_decay, pvals):
        self.net.train()
        client_chosen = (np.arange(len(pvals)) * np.random.multinomial(1, pvals)).sum()
        data = self.trainloader[client_chosen]
        num_data = len(data)
        for it, (inputs, labels) in enumerate(data):
            if it >= num_local_steps:
                break
            # zero the parameter gradients
            self.net.zero_grad()
            # compute the loss
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            for param in self.net.parameters():
                loss += (pvals[client_chosen] / num_data) * weight_decay * torch.norm(param) ** 2
            # compute the gradient of loss with respect to all Tensors with requires_grad=True
            loss.backward()
            # disable gradient calculation to reduce memory consumption
            with torch.no_grad():
                for name, param in self.net.named_parameters():
                    noise = torch.sqrt(2 / lr) * torch.normal(mean=torch.zeros_like(param), std=torch.ones_like(param))
                    conductive_grad = torch.zeros_like(param)
                    for i, (mean_i, precision_i) in enumerate(zip(self.means, self.precisions)):
                        surrogate_i = precision_i[name] * (param - mean_i[name])
                        if i == client_chosen:
                            surrogate_i *= 1 - 1 / pvals[client_chosen]
                        conductive_grad += surrogate_i
                    scaled_grad = num_data / pvals[client_chosen] * param.grad.data + conductive_grad + noise
                    param.data.add_(- lr * scaled_grad)

    def save_results(self, epoch, burnin, thinning, save_samples, path_samples):
        # add the new predictions with the previous ones
        if epoch >= burnin and (
                epoch - burnin) % thinning == 0 and save_samples and path_samples is not None:
            # calculate the accuracy
            acc_test = accuracy_model(self.net, self.testloader, self.device, verbose=False)
            torch.save(copy.deepcopy(self.net).state_dict(), path_samples + '/%s' % epoch)
            # save the accuracy
            self.save_dict["accuracies_test"].append(acc_test)
            self.save_dict["epochs"].append(epoch)
            # print the statistics
            print("--- Test --- Epoch: {}, Test accuracy: {}\n".format(epoch + 1, acc_test))

    def run(self, num_iter, num_local_steps, lr, weight_decay, pvals, burnin = 0, thinning = 1, epoch_init = -1,
            save_samples = False, path_samples = None):
        lr = torch.Tensor([lr]).to(self.device)
        for epoch in tqdm(range(epoch_init + 1, epoch_init + 1 + num_iter)):
            self.net_update(num_local_steps, lr, weight_decay, pvals)
            self.save_results(epoch, burnin, thinning, save_samples, path_samples)
        return self.net.state_dict(), self.save_dict
