import numpy as np


class BaseFald:

    def __init__(self, gradU, param_clients, tau = 1., pc = 1., gamma = 1.):
        self.gradU = gradU
        self.tau, self.pc, self.gamma = tau, pc, gamma
        self.param_clients = np.array(param_clients)
        self.param_server = self.get_server_param()
        self.saved_params = [self.param_server]

    def server_update(self):
        self.param_server = np.mean(self.param_clients, axis=0)
        self.param_clients = np.tile(self.param_server, (len(self.param_clients), *(self.param_clients.ndim - 1) * [1]))
        self.saved_params.append(self.param_server)

    def get_server_param(self):
        return np.mean(self.param_clients, axis=0)

    def get_saved_params(self):
        return np.squeeze(self.saved_params)
