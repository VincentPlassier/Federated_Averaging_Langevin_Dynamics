#!/usr/bin/env python
# coding: utf-8

from itertools import islice

import numpy as np
import torch
from algo_dl.fald_base import BaseFald


class Fald(BaseFald):

    def __init__(self, testloader, trainloader_init, net, param_optimizer, load_init = True, save_init = True,
                 path_server = None):
        super().__init__(testloader, trainloader_init, net, param_optimizer, load_init, save_init, path_server)

    def net_clients_update(self, trainloader, wd, lr):
        running_loss = 0.
        correct = 0
        total = 0
        gaussians = {}
        for name, param in self.net_server.named_parameters():
            gaussians[name] = torch.sqrt(2 / (lr * self.num_clients)) * torch.normal(mean=torch.zeros_like(param),
                                                                                     std=torch.ones_like(param))

        for i, (net, loader) in enumerate(zip(self.net_clients, trainloader)):
            length = len(loader)
            index = np.random.randint(length)
            inputs, targets = list(islice(loader, index, index + 1))[0]
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            net.zero_grad()
            outputs = net(inputs).to(self.device)
            loss = self.criterion(outputs, targets)
            for param in net.parameters():
                loss += wd * torch.norm(param) ** 2
            # compute the gradient of loss with respect to all Tensors with requires_grad=True
            loss.backward()
            running_loss += loss.item()
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            # disable gradient calculation to reduce memory consumption
            with torch.no_grad():
                for name, param in net.named_parameters():
                    scaled_grad = self.num_data * param.grad.data + gaussians[name]
                    # perform the SGD step
                    param.data.sub_(lr * scaled_grad)

    def run(self, trainloader, testloader, num_iter, weight_decay, prob_update_param, lr_dict, pvals, t_burn_in = 0,
            thinning = 1, epoch_init = -1, save_samples = False, path_samples = None):
        self.num_data = 0  # the number of data on the clients
        for loader_i in trainloader:
            self.num_data += sum([len(data[1]) for data in loader_i])
        prob_update_param = prob_update_param['prob_update_param']
        wd = weight_decay / (self.num_clients * self.num_data)
        scheduler = lr_dict['scheduler'](**lr_dict['args'])
        self.net_server_update(pvals)
        for epoch in range(epoch_init + 1, epoch_init + 1 + num_iter):
            scheduler.step()
            lr = torch.Tensor([scheduler.get_lr()]).to(self.device)
            self.net_clients_update(trainloader, wd, lr)
            communication_round = np.random.binomial(1, prob_update_param, 1)
            if communication_round == 1:
                self.net_server_update(pvals)
                self.net_server_transfert()
            self.save_results(testloader, epoch, t_burn_in, thinning, save_samples, path_samples,
                              pvals)
        return self.net_server.state_dict(), self.save_dict
