"""Semi-supervised Variational Autoencoder, often described as M2 (ref. Kingma)

The distribution of the different terms of the model is:
        q(z|x,y) ~ Normal(mu, std), where mu and std are determined by the encoder NN
        p(x|z,y) ~ Normal(mu, std), where mu and std are determined by the decoder NN
        p(z) ~ Normal(0, 1) (Standard Gaussian)
        p(y|x) ~ Normal(mu, std), where mu and std are determined by the regression (NN)
        p(y) ~ Normal(mu, std) where mu and std are calculated beforehand with the script **Insert Name**

Parts of the code is inspired by the exercise in week 7 of the course Deep Learning 02456.

LATEST CHANGES:
    - fixed small errors (sums->mean)
    - introduced batchnorm (helped to avoid large gradients in the NN)
"""



# imports
# TODO: REMOVE UNUSED IMPORTS
from torch import nn, Tensor
import torch
from torch.distributions import Normal, Distribution, Uniform
import h5py
import re
import pandas as pd
import numpy as np
import torch.utils.data
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm
from modules.plotting import *
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_seed(random_seed):
    """
    Function to seed the data-split and backpropagation (to enforce reproducibility)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)




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

class M2(nn.Module):
    def __init__(self, 
                 latent_dim:int,
                 input_dim:int,
                 N_isoform:int,
                 alpha:float,
                 py_data:pd.DataFrame
                 ):
        
        super(M2, self).__init__()

        # this value is used in the loss function. It is defined to be 0.1 * N
        self.alpha = alpha

        self.py_data = py_data
        
        # q(z|x,y)
        # (it takes x and y as inputs)
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim + N_isoform, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2 * latent_dim) 
        )

        # p(x|z,y)
        # (it takes z and y as inputs)
        self.decoder = nn.Sequential( 
            nn.Linear(in_features=latent_dim + N_isoform, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=2 * input_dim),
        )

        # Regression FNN
        # predicts y given x

        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),  
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2*N_isoform)
        )

        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_dim])))



    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # data manipulation
        y = y.to(device)
        x = x.to(device)

        # here it is checked which y's are not available
        # TODO: Make this a function
        index =  torch.sum(y,axis=1)==0

        # splits data in x and y.
        x_unlabelled = x[index]
        x_labelled = x[~index]

        y_labelled = y[~index]


        #* Calculate loss based on these splits
        # LABELLED LOS
        if torch.sum(torch.logical_not(index)) > 1:
            L = self.labelled_loss(x_labelled, y_labelled)
        else:
            L = torch.tensor(0).to(device)
            

        # UNLABELLED LOSS
        if torch.sum(index) > 1:
            U = self.unlabelled_loss(x_unlabelled)
        else:
            U = torch.tensor(0).to(device)
            

        J = L.sum() + U.sum()

        # TODO: write into function
        if torch.sum(torch.logical_not(index)) > 1:
            output_from_regressor = self.regressor(x_labelled)
            y_hat_mu, y_hat_log_sigma = output_from_regressor.chunk(2, dim=-1)

            q_y_x = Normal(y_hat_mu, y_hat_log_sigma.exp())

            log_q_y_x = -reduce(q_y_x.log_prob(y_labelled))
        else:
            log_q_y_x = torch.tensor(0.0).to(device)
        
        J_alpha = J  + self.alpha * log_q_y_x.mean()

        return J_alpha  

        
    def labelled_loss(self, x:Tensor, y:Tensor):
        #* Calculates the labelled loss, denoted L(x,y)
        # encoder now gets both the x and y!
        # TODO: Write this into a function
        input_to_encoder = torch.cat((x,y), dim=1)

        # output of encoder is mu, and log_var!
        output_from_encoder = self.encoder(input_to_encoder)
        # splits:
        mu_encoder, log_sigma_encoder =  output_from_encoder.chunk(2, dim=-1)

        # Now we can build a distribution over all the z's:
        q_z_xy = ReparameterizedDiagonalGaussian(mu_encoder, log_sigma_encoder)
        # (ps this stand for distribution of z given x and y:  q(z|x,y) )

        # then we need the prior over the y's:
        # TODO: insert the distribution over the y's. Should be accessed from py_data that contains the mu and sigma for the normal distribution. 
        #p_y = ReparameterizedDiagonalGaussian(py_data[0], sigma_p_y)
        
        p_y = Normal(torch.tensor(self.py_data['Mean'].values).to(device), 
                     torch.tensor(self.py_data["Standard_Deviation"].values).to(device))

        # PRIOR DISTRIBUTION OVER THE z's !!! They are the same as M1
        # TODO: Write this into a function
        prior_params = self.prior_params.expand(x.size(0), *self.prior_params.shape[-1:])
        mu_prior, log_sigma_prior = prior_params.chunk(2, dim=-1)
        p_z = ReparameterizedDiagonalGaussian(mu_prior, log_sigma_prior)


        # TO APPROXIMATE THE EXPECTATION, WE SAMPLE FROM q_z_xy (the expectation is over q_z_xy)
        z = q_z_xy.rsample()

        # DECODER
        # TODO: write this into a function
        input_to_decoder = torch.cat((z,y), dim=1)
        output_from_decoder = self.decoder(input_to_decoder)
        mu_decoder, log_sigma_decoder = output_from_decoder.chunk(2, dim=1)
        p_x_yz = Normal(mu_decoder, log_sigma_decoder.exp())



        #* AND FOR MY FINAL TRICK...
        # THis idea with reduce is stolen from the original VAE document. 
        # (it just sums up the probability across the non-batch dimension)
        log_p_x_yz = reduce(p_x_yz.log_prob(x))
        log_p_y =  reduce(p_y.log_prob(y))
        # print(log_p_y)
        log_p_z = reduce(p_z.log_prob(z))
        log_q_z_xy = reduce(q_z_xy.log_prob(z))



        L = (-1) * (log_p_x_yz + log_p_y + log_p_z - log_q_z_xy)
        # TODO: think about the minus
        return L


    def unlabelled_loss(self, x:Tensor):
        #* CALCULATES THE UNLABELLED LOSS, DENOTED U(x)

        # LETS FIND THE GAUSSIAN DISTRIBUTION OVER y
        output_from_regressor = self.regressor(x)
        y_hat_mu, y_hat_log_sigma = output_from_regressor.chunk(2, dim=-1)

        # the gaussian distribution over y's
        qy = Normal(y_hat_mu,y_hat_log_sigma.exp())

        # To approximate the expectation, we sample from over regression!!!
        y_hat = qy.sample()

        #* AND FOR MY FINAL TRICK... AGAIN:
        H = reduce(qy.entropy())

        U = (-1) * (-self.labelled_loss(x, y_hat) + H)
        return U



if __name__ == '__main__':

    random_seed_ = 1
    batch_size = 256
    N_isoform = 156958
    latent_dim = 256
    input_dim = 18965
    
    num_epochs = 100
    clipping_value = 0.5
    N_unlabelled = 167884
    learning_rate = 1e-4
    weight_decay = 1e-4


    # seeding 
    random_seed(random_seed_)


    path_to_data = "/dtu-compute/datasets/iso_02456/hdf5-row-sorted/"

    old_path = '/zhome/31/1/155455/DeepLearningProject23/GaussianIsoforms.csv'
    old_path = '/zhome/51/4/167677/Desktop/PROJECT_DL/Final_run_1/GaussianIsoforms.csv'
    py_csv = pd.read_csv(old_path)

    from torch.utils.data import DataLoader
    from modules.MainSplit import MainSplit
    from torch.utils.data import Subset


    # THE MAIN TRAINING SPLIT (some y's are 'labelled')
    train_split = MainSplit(path_to_data, train=True, N_unlabelled=N_unlabelled)
    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)

    alpha = 0.1 * len(train_split)


    # THE MAIN TEST SPLIT (All y's are 'labelled'), Is further split into a training for FNN (called test) and validation 
    test_validation_split = MainSplit(path_to_data, train=False)


    # # splitting (stratified):
    # test_idx, validation_idx = train_test_split(np.arange(len(test_validation_split)),
    #                                             test_size=0.4,
    #                                             random_state=999,
    #                                             shuffle=True,
    #                                             stratify=test_validation_split.tissue_types)



    # test_dataset = Subset(test_validation_split, test_idx)
    # validation_dataset = Subset(test_validation_split, validation_idx)

    # # Testing splits
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    test_loader = DataLoader(test_validation_split, batch_size=batch_size, shuffle=True, drop_last=True)


    # define dictionary to store the training curves
    training_data = defaultdict(list)
    validation_data = defaultdict(list)



    vae = M2(latent_dim, input_dim, N_isoform, alpha=alpha, py_data = py_csv).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Loss function
    criterion = nn.MSELoss()

    # Making sure that things are on CUDA
    vae = vae.to(device)


    

    def removeNaN(t1, t2):
        

        combined_nan_indices =  torch.sum(t1,axis=1)==0

        t1_masked = t1[torch.logical_not(combined_nan_indices)]
        t2_masked = t2[torch.logical_not(combined_nan_indices)]


        return t1_masked, t2_masked

    def train(vae, optimizer, num_epochs):
        epoch = 0
        train_data = defaultdict(list)
        validation_data = defaultdict(list)

        while epoch < num_epochs:
            epoch += 1
            training_epoch_data = defaultdict(list)
            validation_epoch_data = defaultdict(list)
            i = 0
            vae.train()
            for x, y in tqdm(train_loader, desc = "VAE - Training"):
                

                y = y.to(device)
                x = x.to(device)

                optimizer.zero_grad()
                loss = vae(x, y)

                y_hat_mu, y_hat_log_sigma = vae.regressor(x).chunk(2, dim=-1)

                # the gaussian distribution over y's
                qy = Normal(y_hat_mu, y_hat_log_sigma.exp())

                # To approximate the expectation, we sample from over regression!!!
                y_pred = qy.sample()

                y_loss, y_pred_loss = removeNaN(y, y_pred)

                mse_loss = criterion(y_pred_loss, y_loss)

                training_epoch_data["FNN"].append(mse_loss.item())

                loss.backward()

                 # arbitrary value of your choosing
                torch.nn.utils.clip_grad_norm_(vae.parameters(), clipping_value)

                optimizer.step()
                
                training_epoch_data["loss"].append(loss.item())

                if i % 200:
                    #tqdm.write(str(loss.item()))
                    #tqdm.write(str(mse_loss.item()))
                    print(str(np.mean(training_epoch_data["loss"][-100:])))
                    print("number of points:" + str(y_loss.size(0)) + " loss: " + str(np.mean(training_epoch_data["FNN"][-100:])))



                i += 1
                
            train_loss = np.mean(training_epoch_data["loss"])
            train_data["loss"].append(train_loss)

            train_FNN = np.mean(training_epoch_data["FNN"])
            train_data["FNN"].append(train_FNN)

            with torch.no_grad():
                vae.eval()
                i = 0

                for x, y in tqdm(test_loader, desc = "VAE - Validation"):
                    y = y.to(device)
                    x = x.to(device)

                    loss = vae(x,y)

                    y_hat_mu, y_hat_log_sigma = vae.regressor(x.to(device)).chunk(2, dim=-1)

                    # the gaussian distribution over y's
                    qy = Normal(y_hat_mu, y_hat_log_sigma.exp())

                    # To approximate the expectation, we sample from over regression!!!
                    y_pred = qy.sample()

                    y_l, y_pred_l = removeNaN(y, y_pred)

                    mse_loss = criterion(y_pred_l, y_l)

                    validation_epoch_data["FNN"].append(mse_loss.item())
                    validation_epoch_data["loss"].append(loss.item())
                    if i % 100:
                        print(str(loss.item()))
                        print(str(mse_loss.item()))

                    i += 1
                
                validation_loss = np.mean(validation_epoch_data["loss"])
                validation_data["loss"].append(validation_loss)

                validation_FNN = np.mean(validation_epoch_data["FNN"])
                validation_data["FNN"].append(validation_FNN)

                

            with torch.no_grad():    
                createLossPlotFNN(train_data["loss"], validation_data["loss"], "leg")
                createLossPlotFNN(train_data["FNN"], validation_data["FNN"], "leg2")

    train(vae, optimizer, num_epochs)


