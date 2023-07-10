#!/usr/bin/env python
# coding: utf-8

import copy
from collections import OrderedDict
from itertools import islice

import numpy as np
import torch
from algo_dl.fald_base import BaseFald


class VrFaldStar(BaseFald):

    def __init__(self, testloader, trainloader_init, net, param_optimizer, load_init = True, save_init = True,
                 path_server = None):
        super().__init__(testloader, trainloader_init, net, param_optimizer, load_init, save_init, path_server)
        # initialize the variance reduction parameter
        self.net_ref_pt = copy.deepcopy(net)
        self.grad_ref_pt = OrderedDict()

    def net_clients_update(self, trainloader, wd, lr, net_ref_pt):
        running_loss = 0.
        correct = 0
        total = 0
        gaussians = {}
        for name, param in self.net_server.named_parameters():
            gaussians[name] = torch.sqrt(2 / (lr * self.num_clients)) * torch.normal(mean=torch.zeros_like(param),
                                                                                     std=torch.ones_like(param))
        # the i-th worker updates it local parameter
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
            # compute the gradient at the reference point
            net_ref_pt.zero_grad()
            outputs_ref_pt = net_ref_pt(inputs).to(self.device)
            loss_ref_pt = self.criterion(outputs_ref_pt, targets)
            for param_ref_pt in net_ref_pt.parameters():
                loss_ref_pt += wd * torch.norm(param_ref_pt) ** 2
            loss_ref_pt.backward()
            # disable gradient calculation to reduce memory consumption
            with torch.no_grad():
                for (name, param), param_ref_pt in zip(net.named_parameters(), net_ref_pt.parameters()):
                    # compute the gradient transmitted by the i-th worker
                    scaled_grad = self.num_data * (param.grad.data - param_ref_pt.grad.data) + self.grad_ref_pt[name]
                    # perform the SGD step
                    param.data.sub_(lr * (scaled_grad + gaussians[name]))

    @torch.no_grad()
    def ref_pt_update(self, pvals):
        # Parameter updates
        for param_ref_pt in self.net_ref_pt.parameters():
            param_ref_pt.data.copy_(torch.zeros_like(param_ref_pt).to(self.device))
        for p, net in zip(pvals, self.net_clients):
            for param_ref_pt, param_client in zip(self.net_ref_pt.parameters(), net.parameters()):
                param_ref_pt.data.add_(p * param_client)

    def cv_update(self, trainloader, wd):
        for name, param_ref_pt in self.net_ref_pt.named_parameters():
            self.grad_ref_pt[name] = 2 * self.num_data * wd * param_ref_pt
        for loader in trainloader:
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.net_ref_pt.zero_grad()
                outputs = self.net_ref_pt(inputs).to(self.device)
                loss = self.criterion(outputs, targets)
                # compute the gradient of loss with respect to all Tensors with requires_grad=True
                loss.backward()
                # disable gradient calculation to reduce memory consumption
                with torch.no_grad():
                    rescale_factor = len(targets) / self.num_clients
                    for name, param_ref_pt in self.net_ref_pt.named_parameters():
                        self.grad_ref_pt[name] += rescale_factor * param_ref_pt.grad.data

    def run(self, trainloader, testloader, num_iter, weight_decay, prob_communication, lr_dict, pvals, t_burn_in = 0,
            thinning = 1, epoch_init = -1, save_samples = False, path_samples = None):
        prob_update_param, prob_update_cv = prob_communication.values()
        self.num_data = 0  # the number of data on the clients
        for loader_i in trainloader:
            self.num_data += sum([len(data[1]) for data in loader_i])
        wd = weight_decay / (self.num_clients * self.num_data)
        scheduler = lr_dict['scheduler'](**lr_dict['args'])
        self.ref_pt_update(pvals)
        self.cv_update(trainloader, wd)
        self.net_server_update(pvals)
        for epoch in range(epoch_init + 1, epoch_init + 1 + num_iter):
            scheduler.step()
            lr = torch.Tensor([scheduler.get_lr()]).to(self.device)
            communication_round = np.random.binomial(1, prob_update_param, 1)
            cv_update_round = np.random.binomial(1, prob_update_cv, 1)
            net_ref_pt = copy.deepcopy(self.net_ref_pt)
            if cv_update_round:
                self.ref_pt_update(pvals)
            self.net_clients_update(trainloader, wd, lr, net_ref_pt)
            if cv_update_round:
                self.cv_update(trainloader, wd)
            if communication_round == 1:
                # print('\n-- COMMUNICATION --')
                self.net_server_update(pvals)
                self.net_server_transfert()
            self.save_results(testloader, epoch, t_burn_in, thinning, save_samples, path_samples, pvals)
        return self.net_server.state_dict(), self.save_dict
