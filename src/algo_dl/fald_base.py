#!/usr/bin/env python
# coding: utf-8

import copy
import os

import torch
import torch.nn as nn
from tqdm import tqdm
from utils.tools_dl import accuracy_model, client_solver


def init_net_clients(client_data, net, param_optimizer, load_init, save_init, path_save):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # list of the neural networks on each worker
    net_clients = []
    if param_optimizer['num_epochs'] == 0:
        print('\n--- No initialization. ---')
    print("-> path_save:", path_save)
    for i, data in tqdm(enumerate(client_data)):
        net_client = copy.deepcopy(net).to(device)
        path_client = os.path.join(path_save, f'client_{i}')
        if load_init and path_save is not None and os.path.exists(path_client):
            # load the model of the client
            net_client.load_state_dict(torch.load(path_client))
        elif param_optimizer['num_epochs'] > 0:
            # perform some SGD steps
            model_state_dict = client_solver(data, net_client, **param_optimizer)
            if save_init and path_save is not None:
                # store the model of the client
                torch.save(model_state_dict, path_client)
        # store the neural network
        net_clients.append(copy.deepcopy(net_client).to(device))
    return net_clients


class BaseFald:

    def __init__(self, testloader, trainloader_init, net, param_optimizer, load_init = True, save_init = True,
                 path_server = None, criterion = nn.CrossEntropyLoss(reduction='mean')):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_clients = len(trainloader_init)
        self.num_data = None
        self.criterion = criterion
        self.net_server = net
        print('Accuracy before init:',
              accuracy_model(self.net_server, testloader, self.device, verbose=False))
        self.net_clients = init_net_clients(trainloader_init, net, param_optimizer, load_init, save_init, path_server)
        print('Accuracy after init:',
              accuracy_model(self.net_server, testloader, self.device, verbose=False))
        # Store the statistics
        self.save_dict = {"epochs": [], "losses_test": [], "accuracies_test": [], "mse_relative": []}

    @torch.no_grad()
    def net_server_update(self, pvals):
        # Parameter updates
        for param_server in self.net_server.parameters():
            param_server.data.copy_(torch.zeros_like(param_server).to(self.device))
        for p, net_client in zip(pvals, self.net_clients):
            net = copy.deepcopy(net_client)
            for param_server, param in zip(self.net_server.parameters(), net.parameters()):
                param_server.data.add_(p * param)

    @torch.no_grad()
    def net_server_transfert(self):
        # the server transmit the global parameter to the clients
        for net in self.net_clients:
            net.load_state_dict(copy.deepcopy(self.net_server).state_dict())

    def save_results(self, testloader, epoch, t_burn_in, thinning, save_samples, path_save_samples, pvals):
        # add the new predictions with the previous ones
        if epoch >= t_burn_in and (
                epoch - t_burn_in) % thinning == 0 and save_samples and path_save_samples is not None:
            self.net_server_update(pvals)
            # calculate the accuracy
            acc_test = accuracy_model(copy.deepcopy(self.net_server), testloader, self.device, verbose=False)
            torch.save(copy.deepcopy(self.net_server).state_dict(), path_save_samples + '/%s' % epoch)
            # Save the parameters of each client
            for i, net_client in enumerate(self.net_clients):
                path_client = os.path.join(path_save_samples, f'client_{i}')
                torch.save(copy.deepcopy(net_client).state_dict(), path_client)
            # save the accuracy
            self.save_dict["accuracies_test"].append(acc_test)
            self.save_dict["epochs"].append(epoch)
            # print the accuracy
            print("--- Test --- Epoch: {}, Test accuracy: {}\n".format(epoch + 1, acc_test))
