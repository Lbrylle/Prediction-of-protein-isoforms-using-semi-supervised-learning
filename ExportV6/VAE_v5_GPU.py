import math
import torch
import modules.IsoDatasets as IsoDatasets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from modules.plotting import *
from tqdm import tqdm
from collections import defaultdict
from typing import *
from functools import reduce
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.nn.parameter import Parameter
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.distributions import Distribution
from torch.distributions import Normal
from torch.distributions import Bernoulli
from sklearn.model_selection import StratifiedShuffleSplit
from modules.StratifiedSampling import StratifiedSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ", device)

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

    def __init__(self, batch_cols, input_shape:torch.Size, latent_features:int, dropout_rate) -> None:
        super(VariationalAutoencoder, self).__init__()
        self.input_shape = (batch_cols, )
        self.latent_features = latent_features
        self.observation_features =  batch_cols
        
        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.observation_features, out_features=8192),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=8192, out_features=4096),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=4096, out_features=2048),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=2048, out_features=2 * latent_features) 
        )

        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_features, out_features=2048),
            nn.Tanh(),
            nn.Linear(in_features=2048, out_features=4096),
            nn.Tanh(),
            nn.Linear(in_features=4096, out_features=8192),
            nn.Tanh(),
            nn.Linear(in_features=8192, out_features=self.observation_features),
        )

        # Regression FNN
        self.regressor = nn.Sequential(
            nn.Linear(latent_features, 8),  # Reduce the number of hidden units
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(8, 4),  # Further reduce if necessary
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4, 156958)
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
        qz = self.posterior(x)
        
        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()

        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)

        # regressor
        z_r = self.regressor(z)

        return {'px': px, 'pz': pz, 'qz': qz, 'z': z, 'z_r': z_r}

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

        # Forward pass
        outputs = model(x)
        
        # Unpack outputs
        px, pz, qz, z, z_r = [outputs[k] for k in ["px", "pz", "qz", "z", "z_r"]]
        
        # evaluate log probabilities
        log_px = reduce(px.log_prob(x))
        log_pz = reduce(pz.log_prob(z))
        log_qz = reduce(qz.log_prob(z))
        

        # compute the ELBO with and without the beta parameter:
        # `L^\beta = E_q [ log p(x|z) ] - \beta * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl = log_qz - log_pz
        elbo = log_px - kl 
        beta_elbo = log_px - self.beta * kl

        loss = -beta_elbo.mean()

        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px':log_px, 'kl': kl}

        return loss, diagnostics, outputs

batch_size = 64

# Creation of GtexDataset instances
gtex = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/")



# Stratified Sampling for train and val
train_idx, validation_idx = train_test_split(np.arange(len(gtex)),
                                             test_size=0.1,
                                             random_state=999,
                                             shuffle=True,
                                             stratify=gtex.tissue_types)

# Subset dataset for train and val
train_dataset = Subset(gtex, train_idx)
validation_dataset = Subset(gtex, validation_idx)

# Dataloader for train and val
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

# define dictionary to store the training curves
training_data = defaultdict(list)
validation_data = defaultdict(list)

# VAE params
latent_features = 512
vae = VariationalAutoencoder(18965, batch_size, latent_features, dropout_rate = 0.5)

# VI params
beta = 1.0
vi = VariationalInference(beta=beta)

# Optimizer
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3, weight_decay=1e-4)

# Loss function
criterion = nn.MSELoss()

# Making sure that things are on CUDA
vae = vae.to(device)
vi = vi.to(device)

# Training params
num_epochs = 50


def train(vi, vae, optimizer, num_epochs, device):
    epoch = 0
    train_data = defaultdict(list)
    validation_data = defaultdict(list)

    while epoch < num_epochs:
        epoch += 1
        training_epoch_data = defaultdict(list)
        validation_epoch_data = defaultdict(list)

        vae.train()
        for x, y, tissue in tqdm(train_loader, desc = "VAE - Training"):
            
            optimizer.zero_grad()
            _, diagnostics, outputs = vi(vae, x)

            y_pred = outputs["z_r"]
            y_pred = y_pred.squeeze()

            loss = criterion(y_pred, y)

            loss.backward()
            optimizer.step()
            
            training_epoch_data["loss"].append(loss.item())
            
        train_loss = np.mean(training_epoch_data["loss"])
        train_data["loss"].append(train_loss)

        with torch.no_grad():
            vae.eval()

            for x, y, tissue in tqdm(validation_loader, desc = "VAE - Validation"):
                _, diagnostics, outputs = vi(vae, x)

                y_pred = outputs["z_r"]
                y_pred = y_pred.squeeze()

                loss = criterion(y_pred, y)

                validation_epoch_data["loss"].append(loss.item())
            
            validation_loss = np.mean(validation_epoch_data["loss"])
            validation_data["loss"].append(validation_loss)

        with torch.no_grad():    
            createLossPlotFNN(train_data["loss"], validation_data["loss"], "Loss")
        
train(vi, vae, optimizer, num_epochs, device)