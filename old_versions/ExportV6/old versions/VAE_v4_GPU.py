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

    def __init__(self, batch_cols,input_shape:torch.Size, latent_features:int) -> None:
        super(VariationalAutoencoder, self).__init__()
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

        # Regression FNN
        self.regressor = nn.Sequential(
            nn.Linear(latent_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 156958)  # Output dimension for regression
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

        regression_output = self.regressor(z)

        return {'px': px, 'pz': pz, 'qz': qz, 'z': z, 'regression_output': regression_output}


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

# Create your GtexDataset instances
gtex_train = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/")
gtex_test = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/")

# Use the StratifiedSampler in your DataLoader
gtx_train_dataloader = DataLoader(gtex_train, batch_size=64)
gtx_test_dataloader = DataLoader(gtex_test, batch_size=64)

# VAE
latent_features = 256
vae = VariationalAutoencoder(18965, batch_size, latent_features)

# Evaluator: Variational Inference
beta = 1.0
vi = VariationalInference(beta=beta)

# FNN definition
num_classes = 156958
num_hidden = 256
criterion = nn.MSELoss()

# The Adam optimizer works really well with VAEs.
optimizerVAE = torch.optim.Adam(vae.parameters(), lr=1e-2)

# define dictionary to store the training curves
training_data = defaultdict(list)
validation_data = defaultdict(list)

# move the model to the device
vae = vae.to(device)
vi = vi.to(device)

cur_loss = []

def train(vae, vi, optimizerVAE, num_epochs, device):
    epoch = 0
    FNN_train_loss, FNN_eval_loss = [], []
    
    while epoch < num_epochs:
        epoch += 1
        training_epoch_data = defaultdict(list)
        vae.train()
        cur_loss = []

        for x, y, tissue in tqdm(gtx_train_dataloader, desc="VAE - Training"):
            print(tissue)
            x = x.to(device)
            y = y.to(device)

            optimizerVAE.zero_grad()

            # Forward pass through VAE and obtain VAE loss
            vae_outputs = vi(vae, x)
            vae_loss, diagnostics, outputs = vae_outputs
            cur_loss.append(vae_loss.item())  # Store VAE loss for monitoring
            
            # Get the regression output
            regression_output = outputs['regression_output']

            # Compute FNN loss
            y_pred = regression_output.squeeze()  # Assuming a single output for regression
            FNN_loss = criterion(y_pred, y)  # Assuming using MSE Loss for regression

            # Combine VAE and FNN losses
            total_loss = vae_loss + FNN_loss
            total_loss.backward()
            
            optimizerVAE.step()

            # Gather data for the current batch
            for k, v in diagnostics.items():
                training_epoch_data[k] += [v.mean().item()]

        FNN_train_loss.append(np.mean(cur_loss))

        for k, v in training_epoch_data.items():
            training_data[k] += [np.mean(training_epoch_data[k])]

        # Validation step
        with torch.no_grad():
            vae.eval()

            x_val, y_val, tissue_val = next(iter(gtx_test_dataloader))
            x_val = x_val.to(device)
            y_val = y_val.to(device)

            # Forward pass through VAE for validation
            val_loss, val_diagnostics, val_outputs = vi(vae, x_val)
            val_regression_output = val_outputs['regression_output']
            val_y_pred = val_regression_output.squeeze()  # Assuming a single output for regression

            val_FNN_loss = criterion(val_y_pred, y_val)  # Assuming using MSE Loss for regression

            FNN_eval_loss.append(val_FNN_loss.item())

            # Gather data for the validation step
            for k, v in val_diagnostics.items():
                validation_data[k] += [v.mean().item()]    
        
        # Print or visualize losses if needed
        print(FNN_train_loss, FNN_eval_loss)
        createLossPlotFNN(FNN_train_loss, FNN_eval_loss, "FNN")


        # Reproduce the figure from the beginning of the notebook, plot the training curves and show latent samples
        with torch.no_grad():
            createELBOPlotVAE(training_data, validation_data, epoch)
            createLossPlotVAE(-np.array(training_data['log_px']) + beta * np.array(training_data['kl']), 
                            -np.array(validation_data['log_px']) + beta * np.array(validation_data['kl']), epoch)

    
num_epochs = 200
train(vae, vi, optimizerVAE, num_epochs, device)
