# import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

from algo_np.fald_base_np import BaseFald
from utils.prior_likelihood import Gaussian


class Fald(BaseFald):
    """
    Fald.
    """

    def __init__(self, gradU, param_clients, tau = 1., pc = 1., gamma = 1.):
        super().__init__(gradU, param_clients, tau, pc, gamma)

    def param_update(self):
        num_clients, dim = self.param_clients.shape[0], self.param_clients.shape[1:]
        gauss_common = np.random.randn(*dim)
        gaussians = np.random.randn(num_clients, *dim)
        for i, g in enumerate(gaussians):
            noise = np.sqrt(2 * self.gamma) * (
                    np.sqrt(self.tau / num_clients) * gauss_common + np.sqrt(1 - self.tau) * g)
            self.gradU[i].update_minibatch()
            self.param_clients[i] += - self.gamma * self.gradU[i].gradsto(self.param_clients[i]) + noise

    def step(self):
        self.param_update()
        bernouilli = np.random.binomial(1, self.pc, 1)
        if bernouilli == 1:
            self.server_update()
            return True


if __name__ == '__main__':
    # fix the seed for reproducibility
    np.random.seed(20)
    # define the dimension
    dim = 1
    # set the number of workers
    num_clients = 5
    # initialize the parameters
    param_clients = np.zeros((num_clients, dim))
    # contain the potentials of each workers
    gradU = list()
    # set the learning rate of the clients
    gamma = .1
    #
    mu, cov = np.zeros(dim), np.identity(dim)
    gauss = Gaussian(mu, num_clients * cov)
    #
    for i in range(num_clients):
        gradU.append(gauss.minus_grad_log)
    # set the privacy parameter
    tau = .99
    # probability of communication
    pc = .95
    # define the Fald sampler
    fald = Fald(gradU, param_clients, tau, pc, gamma)
    # define the number of iterations
    mc_iter = 10000
    # run the algorithm
    for _ in range(mc_iter):
        fald.step()
    param_list = fald.get_saved_params()
    # Display the results
    plt.hist(np.squeeze(param_list), 30, density=True)
    X = np.linspace(-3, 3, 500)
    plt.plot(X, [ss.multivariate_normal.pdf(x, mean=mu, cov=cov) for x in X])
    plt.grid('True')
    plt.title('-K={0:.0E}, gamma={1:.1E}, tau={2:.1E}'.format(mc_iter, gamma, tau))
    # plt.savefig('figures/' + os.path.basename(__file__)[:-3] +
    #             '-K={0:.0E}, gamma={1:.1E}, tau={2:.1E}'.format(mc_iter, gamma, tau) + '.pdf', bbox_inches='tight')
    plt.show()
