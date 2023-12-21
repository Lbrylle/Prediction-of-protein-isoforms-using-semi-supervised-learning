"""Un-supervised Variational Autoencoder, often described as M1 (ref. Kingma)

The distribution of the different terms of the model is:
        q(z|x) ~ Normal(mu, std), where mu and std are determined by the encoder NN
        p(x|z) ~ Normal(mu, std), where mu and std are determined by the decoder NN
        p(z) ~ Normal(0, 1) (Standard Gaussian)

Parts of the code is inspired by the exercise in week 7 of the course Deep Learning 02456.
"""



# imports
import torch
import matplotlib.pyplot as plt
import numpy as np
from modules.plotting import * # import a plotting function for showing temporary results
from modules.helperFunctions import * # importing reduce, ReparameterizedDiagonalGaussian, init_dataloaders
from tqdm import tqdm
from collections import defaultdict
from typing import *
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.distributions import Distribution
from sklearn.model_selection import train_test_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ", device)


############################################################################################################
#* THE MAIN MODEL (M1) (largely inspired by week 7 in deep learning)
############################################################################################################


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
            nn.Linear(in_features=self.observation_features, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
            nn.Linear(in_features=256, out_features=2*latent_features) # <- note the 2*latent_features
        )

        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=2*self.observation_features),
        )

        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(x|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        h_x = self.encoder(x)
        mu, log_sigma =  h_x.chunk(2, dim=-1)
        
        # return a distribution `q(x|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma, device)

    def prior(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)

        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma, device)

    def observation_model(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""
        mu, log_sigma = self.decoder(z).chunk(2, dim=-1)

        return ReparameterizedDiagonalGaussian(mu, log_sigma, device)

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



if __name__ == '__main__':
    # parameters:
    batch_size = 64
    latent_features = 512
    num_epochs = 100
    experiment_name = 'final_run_for_vae'
    beta = 1.0
    output_dim = 156958
    lr = 1e-4
    input_dim = 18965
    random_seed_ = 1



    # seeding 
    random_seed(random_seed_)

    # save_folders
    save_path = init_folders(experiment_name)

    # save the parameters in the folder
    save_parameters(save_path,
                    batch_size=batch_size,
                    latent_features=latent_features,
                    num_epochs=num_epochs,
                    experiment_name=experiment_name,
                    beta=beta,
                    output_dim=output_dim,
                    lr=lr,
                    input_dim=input_dim,
                    random_seed_=random_seed_
                    )


    ############################################################################################################
    #* THE REGRESSOR
    ############################################################################################################

    regressor = nn.Sequential(
                nn.Linear(latent_features, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, output_dim)  # Output dimension for regression
            ).to(device)




    ############################################################################################################
    #* LOADING DATA (MainSplit)
    ############################################################################################################

    path_to_data = "/dtu-compute/datasets/iso_02456/hdf5-row-sorted/"


    from torch.utils.data import DataLoader
    from modules.MainSplit import MainSplit
    from torch.utils.data import Subset


    # THE MAIN TRAINING SPLIT (some y's are 'labelled')
    train_split = MainSplit(path_to_data, train=True)
    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)


    # THE MAIN TEST SPLIT (All y's are 'labelled'), Is further split into a training for FNN (called test) and validation 
    test_validation_split = MainSplit(path_to_data, train=False)


    # splitting (stratified):
    test_idx, validation_idx = train_test_split(np.arange(len(test_validation_split)),
                                                test_size=0.4,
                                                random_state=999,
                                                shuffle=True,
                                                stratify=test_validation_split.tissue_types)


    test_dataset = Subset(test_validation_split, test_idx)
    validation_dataset = Subset(test_validation_split, validation_idx)

    # Testing splits
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True)



    ############################################################################################################
    #* TRAINING THE MODEL
    ############################################################################################################

    # VAE
    vae = VariationalAutoencoder(input_dim, batch_size, latent_features)

    # Evaluator: Variational Inference

    vi = VariationalInference(beta=beta)

    # FNN definition

    criterion = nn.MSELoss()

    # The Adam optimizer works really well with VAEs.
    optimizerVAE = torch.optim.Adam(vae.parameters(), lr=lr)
    optimizerFNN = torch.optim.Adam(regressor.parameters(), lr=lr)

    # define dictionary to store the training curves
    training_data = defaultdict(list)
    validation_data = defaultdict(list)

    # move the model to the device
    vae = vae.to(device)
    vi = vi.to(device)




    save_data = defaultdict(list)


    def train_vae(vae, 
                vi, 
                train_data_loader, 
                test_data_loader,
                optimizer, 
                num_epochs, 
                device):
        save_data = defaultdict(list)

        epoch = 0
        vae.train()
        with tqdm(total=num_epochs * (len(train_data_loader) + len(test_data_loader)), desc='VAE - Training') as pbar:
            while epoch < num_epochs:
                epoch += 1
                epoch_data = defaultdict(list)
                
                
                
                vae.train()
                for i, (x, _ ) in enumerate(train_data_loader):
                    x = x.to(device)
                    #y = y.to(device)

                    # Forward pass through VAE and obtain VAE loss
                    vae_loss, diagnostics, outputs = vi(vae, x)
                    
                    optimizer.zero_grad()
                    vae_loss.backward()

                    clipping_value = 0.5 # arbitrary value of your choosing
                    torch.nn.utils.clip_grad_norm_(vae.parameters(), clipping_value)


                    optimizer.step()

                    epoch_data['vae_trainloss'].append(vae_loss.item())

                    pbar.update(1)

                    if i % 100 == 0:
                        tqdm.write("VAE_trainloss: " + str(vae_loss.item()))
                
                vae.eval()
                for i, (x, _) in enumerate(test_data_loader):
                    x = x.to(device)

                    vae_loss, diagnostics, outputs = vi(vae, x)

                    epoch_data['vae_testloss'].append(vae_loss.item())

                    

                
                #plot_temp({'vae_loss': epoch_data['vae_loss']}, save_path + '/m1_loss' + str(epoch) + '.png')
                save_data['vae_trainloss'].append(np.mean(epoch_data['vae_trainloss']))
                save_data['vae_testloss'].append(np.mean(epoch_data['vae_testloss']))

    
                np.save(save_path + '/m1_trainloss.npy', save_data['vae_trainloss'])
                np.save(save_path + '/m1_testloss.npy', save_data['vae_testloss'])

                torch.save(vae.encoder.state_dict(), save_path + '/encoder.pth')
                #torch.save(vae.decoder.state_dict(), save_path + '/decoder.pth')


        

        
    def train_fnn(vae,
                vi,
                optimizer, 
                regressor, 
                train_data_loader,
                test_data_loader,
                num_epochs, 
                device,
                criterion):
        epoch = 0
        vae.train()
        
        save_data = defaultdict(list)
        with tqdm(total=num_epochs * (len(train_data_loader) + len(test_data_loader)), desc='FNN - Training') as pbar:
            while epoch < num_epochs:
                epoch += 1
                epoch_data = defaultdict(list)
                regressor.train()
                for i, (x, y) in enumerate(train_data_loader):
                    x = x.to(device)
                    y = y.to(device)

                    latent = vae(x)['z']


                    y_pred = regressor(latent)

                    loss = criterion(y_pred, y)


                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_data['fnn_trainloss'].append(loss.item())

                    pbar.update(1)

                    if i % 100:
                        tqdm.write("FNN_trainloss: " + str(loss.item()))
                


                
                regressor.eval()
                for i, (x,y) in enumerate(test_data_loader):
                    
                    x = x.to(device)
                    y = y.to(device)


                    latent = vae(x)['z']   
                    y_pred = regressor(latent)



                    loss = criterion(y_pred, y)
                    optimizer.step()

                    epoch_data['fnn_testloss'].append(loss.item())

                    pbar.update(1)

                    if i % 100:
                        tqdm.write("FNN_testloss: " + str(loss.item()))

            

                save_data['fnn_trainloss'].append(np.mean(epoch_data['fnn_trainloss']))
                save_data['fnn_testloss'].append(np.mean(epoch_data['fnn_testloss']))
                
                np.save(save_path + 'fnntest_last_epoch.npy', epoch_data['fnn_testloss'])

                np.save(save_path + 'm1_trainfnn.npy', save_data['fnn_trainloss'])
                np.save(save_path + 'm1_testfnn.npy', save_data['fnn_testloss'])

                torch.save(regressor.state_dict(), save_path + '/regressor.pth')






    train_vae(vae, 
            vi, 
            train_loader, 
            test_loader,
            optimizerVAE, 
            num_epochs, 
            device)


    train_fnn(vae,
            vi,
            optimizerFNN, 
            regressor, 
            test_loader,
            validation_loader, 
            num_epochs, 
            device, 
            criterion)


