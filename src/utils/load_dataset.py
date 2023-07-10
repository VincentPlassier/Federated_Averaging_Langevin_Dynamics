#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
import torch
from torch.utils.data import TensorDataset


def load_CIFAR10(root, train = False, transform = torch.Tensor):
    if train:
        path_x = os.path.join(root, "cifar10_train_x.csv")
        path_y = os.path.join(root, "cifar10_train_y.csv")
    else:
        path_x = os.path.join(root, "cifar10_test_x.csv")
        path_y = os.path.join(root, "cifar10_test_y.csv")

    x = np.loadtxt(path_x)
    x = x.reshape((len(x), 32, 32, 3))
    x_stack = []
    for xx in map(transform, x):
        x_stack.append(xx)
    x = torch.stack(x_stack).float()
    y = np.loadtxt(path_y).astype(int)
    return TensorDataset(x.reshape((len(x), 3, 32, 32)), torch.from_numpy(y))
