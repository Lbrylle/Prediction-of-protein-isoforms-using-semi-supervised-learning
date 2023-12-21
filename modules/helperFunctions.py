"""Helper functions for seeding and saving data and more.
"""


import torch
import numpy as np
import time
import os
from torch import Tensor
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def save_parameters(save_path, **kwargs):
    with open(os.path.join(save_path,  "parameters.txt"), 'w') as f:  
        for key, value in kwargs.items():  
            f.write('%s:%s\n' % (key, value))


def random_seed(random_seed):
    """
    Function to seed the data-split and backpropagation (to enforce reproducibility)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)



def init_folders(experiment_name):
    """
    Create folders for saving data.
    """
    main_path = os.getcwd()
    today = time.strftime("%d_%m")

    main_folder = os.path.join(main_path, 'results/')
    if not os.path.exists(main_folder):
        os.mkdir(main_folder)
    if not os.path.exists(main_folder + today):
        os.mkdir(main_folder + today)
    if not os.path.exists(main_folder + today + '/' + experiment_name):
        os.mkdir(main_folder + today + '/' + experiment_name)
    save_path = main_folder + today + '/' + experiment_name + '/'

    return save_path



class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """
    def __init__(self, mu: Tensor, log_sigma:Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()

    def sample_epsilon(self) -> Tensor:
        """`\eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_().to(device)

    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()

    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        eps = torch.normal(0.0, 1.0, self.mu.shape).to(device)

        z = (self.mu + self.sigma * eps).to(device)
        return z

    def log_prob(self, z:Tensor) -> Tensor:
        """return the log probability: log `p(z)`"""
        m = Normal(self.mu, self.sigma)
        return m.log_prob(z).to(device)

def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)