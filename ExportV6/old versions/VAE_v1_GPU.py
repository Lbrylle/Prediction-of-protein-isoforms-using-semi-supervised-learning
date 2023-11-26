import math
import torch
import modules.IsoDatasets 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
#import seaborn as snsc
#import torch.utils.data
from modules.plotting import createLossPlotVAE, createELBOPlotVAE

from tqdm import tqdm
from collections import defaultdict
from typing import *
from functools import reduce
#from IPython.display import Image, display, clear_output
from torch import nn, Tensor
from torch.utils.data import DataLoader
#from torch.utils.data.sampler import SubsetRandomSampler
#from torchvision.transforms import ToTensor
#from torch.nn.functional import softplus
from torch.distributions import Distribution
from torch.distributions import Normal
from torch.distributions import Bernoulli
#from plotting import make_vae_plots

#device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.set_default_device(device)
print("hej", device)

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

class VariationalAutoencoder(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """

    def __init__(self, batch_cols,input_shape:torch.Size, latent_features:int) -> None:
        super(VariationalAutoencoder, self).__init__()
        print(input_shape)
        self.input_shape = (batch_cols, )
        self.latent_features = latent_features
        self.observation_features =  batch_cols
        
        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.observation_features, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=64),
            nn.Tanh(),
            # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
            nn.Linear(in_features=64, out_features=2*latent_features) # <- note the 2*latent_features
            
        )

        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_features, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=self.observation_features),
        )

        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))

    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(x|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        h_x = self.encoder(x)
        mu, log_sigma =  h_x.chunk(2, dim=-1)
        
        # return a distribution `q(x|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def prior(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)

        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def observation_model(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""
        px_logits = self.decoder(z)
        
        px_logits = px_logits.view(-1, *self.input_shape) # reshape the output
        return Bernoulli(logits=px_logits, validate_args=False)

    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""
        
        # flatten the input
        x = x.view(x.size(0), -1)
        
        # define the posterior q(z|x) / encode x into q(z|x)
        print(self.observation_features)
        qz = self.posterior(x)
        
        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()

        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)

        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}


    def sample_from_prior(self, batch_size:int=100):
        """sample z~p(z) and return p(x|z)"""

        # degine the prior p(z)
        pz = self.prior(batch_size=batch_size)

        # sample the prior
        z = pz.rsample()

        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)

        return {'px': px, 'pz': pz, 'z': z}

def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)

class VariationalInference(nn.Module):
    def __init__(self, beta:float=1.):
        super().__init__()
        self.beta = beta

    def forward(self, model:nn.Module, x:Tensor) -> Tuple[Tensor, Dict]:

        # forward pass through the model
        
        outputs = model(x)
        
        # unpack outputs
        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]
        
        # evaluate log probabilities
        log_px = reduce(px.log_prob(x))
        log_pz = reduce(pz.log_prob(z))
        log_qz = reduce(qz.log_prob(z))

        # compute the ELBO with and without the beta parameter:
        # `L^\beta = E_q [ log p(x|z) ] - \beta * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl = log_qz - log_pz
        elbo = log_px - kl # <- your code here
        beta_elbo = log_px - self.beta * kl

        # loss
        loss = -beta_elbo.mean()

        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px':log_px, 'kl': kl}

        return loss, diagnostics, outputs

batch_size = 64

gtex_train = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/", exclude='brain')
gtex_test = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/" , include='brain')

print("gtex training set size:", len(gtex_train))
print("gtex test set size:", len(gtex_test))

gtx_train_dataloader = DataLoader(gtex_train, batch_size=64, shuffle=True)
gtx_test_dataloader = DataLoader(gtex_test, batch_size=64, shuffle=True)


# define the models, evaluator and optimizer

# VAE
latent_features = 2
vae = VariationalAutoencoder(18965, batch_size, latent_features)

# Evaluator: Variational Inference
beta = 1.0
vi = VariationalInference(beta=beta)

# The Adam optimizer works really well with VAEs.
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# define dictionary to store the training curves
training_data = defaultdict(list)
validation_data = defaultdict(list)

epoch = 0

num_epochs = 100

# move the model to the device
vae = vae.to(device)
vi = vi.to(device)

while epoch < num_epochs:
    epoch+= 1
    training_epoch_data = defaultdict(list)
    vae.train()

    # Go through each batch in the training dataset using the loader
    # Note that y is not necessarily known as it is here
    for x, y in tqdm(gtx_train_dataloader):
        # perform a forward pass through the model and compute the ELBO

        train_loss, diagnostics, outputs = vi(vae, x)
        
        optimizer.zero_grad()
        train_loss.backward()
        print('train_loss', train_loss)
        optimizer.step()

        # gather data for the current bach
        for k, v in diagnostics.items():
            training_epoch_data[k] += [v.mean().item()]

    # gather data for the full epoch
    for k, v in training_epoch_data.items():
        training_data[k] += [np.mean(training_epoch_data[k])]

    # Evaluate on a single batch, do not propagate gradients
    with torch.no_grad():
        vae.eval()

        # Just load a single batch from the test loader
        x, y = next(iter(gtx_test_dataloader))
        x = x.to(device)
        #y = y.to(device)

        # perform a forward pass through the model and compute the ELBO
        valid_loss, diagnostics, outputs = vi(vae, x)
        print('valid_loss', valid_loss)

        # gather data for the validation step
        for k, v in diagnostics.items():
            validation_data[k] += [v.mean().item()]
    
    # Reproduce the figure from the begining of the notebook, plot the training curves and show latent samples
    with torch.no_grad():
        createELBOPlot(training_data, validation_data, epoch)
        createLossPlot(- np.array(training_data['log_px']) + beta * np.array(training_data['kl']), 
                        - np.array(validation_data['log_px']) + beta * np.array(validation_data['kl']), epoch)
