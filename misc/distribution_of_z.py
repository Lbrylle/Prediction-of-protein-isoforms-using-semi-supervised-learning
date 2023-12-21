"""local test of z distribution

sampling a labelled datapoint and encode it with the m1 and m2
"""


# imports. 
from sklearn.metrics import r2_score
from torch import nn, Tensor
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from modules.helperFunctions import ReparameterizedDiagonalGaussian, init_folders, random_seed
import matplotlib.pyplot as plt
import os
import time
import random

if True: # dette er et eksperiment med nogle scienceplot-pakke
    import scienceplots
    plt.style.use('science')
    plt.rcParams.update({'figure.dpi': '200'})
    plt.rcParams.update({"legend.frameon": True})
    plt.rcParams.update({'font.size': 17})
    plt.rcParams.update({'font.weight': 'bold'})



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




def init_dataloaders(batch_size):
    from torch.utils.data import DataLoader
    from modules.MainSplit import MainSplit
    from torch.utils.data import Subset


    # THE MAIN TRAINING SPLIT (some y's are 'labelled')
    train_split = MainSplit(path_to_data, train=True)
    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)



    # THE MAIN TEST SPLIT (All y's are 'labelled'), Is further split into a training for FNN (called test) and validation 
    test_validation_split = MainSplit(path_to_data, train=False, latent=True)


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

    return train_loader, test_loader, validation_loader




# parameters
latent_features = 512
output_dim = 156958
input_dim = 18965
batch_size = 1
random_seed_ = 1

experiment_name = 'latent_plots'

save_path = init_folders(experiment_name)

path_to_data = "/dtu-compute/datasets/iso_02456/hdf5-row-sorted/"

random_seed(random_seed_)



############################################################################################################
#* LOAD M2
############################################################################################################


N_isoform = 156958
latent_dim = 512
input_dim = 18965


encoder_m2 = nn.Sequential(
    nn.Linear(in_features=input_dim + N_isoform, out_features=256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=2 * latent_dim) 
).to(device)

encoder_m2_pth = '/zhome/51/4/167677/Desktop/PROJECT_DL/results/19_12/m2/encoder2.pth'

encoder_m2.load_state_dict(torch.load(encoder_m2_pth))
encoder_m2.eval()

def m2_encode(x, y):
    input_to_encoder = torch.cat((x,y), dim=1)

    output_from_encoder = encoder_m2(input_to_encoder)

    mu_encoder, log_sigma_encoder =  output_from_encoder.chunk(2, dim=-1)

    # Now we can build a distribution over all the z's:
    q_z_xy = ReparameterizedDiagonalGaussian(mu_encoder, log_sigma_encoder, device)

    return q_z_xy.rsample()



############################################################################################################
#* LOAD M1 BASED FFN & ENCODER
############################################################################################################

encoder_pth = '/zhome/51/4/167677/Desktop/PROJECT_DL/results/19_12/m1_final_run_relu_256_no_batchnorm/encoder.pth'

encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=2*latent_features) 
        ).to(device)
    
encoder.load_state_dict(torch.load(encoder_pth))
encoder.eval()


def encode(x):
    h_x = encoder(x)
    mu, log_sigma =  h_x.chunk(2, dim=-1)
    
    return ReparameterizedDiagonalGaussian(mu, log_sigma, device).rsample()



def createLatentPlot(z, save_path, name, title, title2):

    def gauss_pdf(x):
        return 1/(np.sqrt(2*np.pi)) * np.exp(-1/2*(x)**2)

    xs = np.linspace(-5, 5, 100)
    y = gauss_pdf(xs)
    plt.figure(figsize=(6,6))
    plt.hist(z, bins=50, density=True)
    plt.plot(xs, y, 'r--')
    plt.title(f'ID: {title}. Model: {title2}')
    plt.xlabel(r'$z$')
    plt.ylabel('Frequency')
    plt.tight_layout()
    tmp_img = save_path + name + '.png'
    plt.savefig(tmp_img)
    plt.close()


############################################################################################################
#* LOAD DATA
############################################################################################################

train_loader, test_loader, validation_loader = init_dataloaders(batch_size)

N_images = 10

for i, (x,y, id_) in enumerate(validation_loader):
    z = encode(x.to(device)).flatten().detach().cpu().numpy()
    createLatentPlot(z, save_path, f'latent_{id_.item()}', id_.item(), 'M1')

    z_m2 = m2_encode(x.to(device), y.to(device)).flatten().detach().cpu().numpy()
    createLatentPlot(z_m2, save_path, f'latent_m2_{id_.item()}', id_.item(), 'M2')
    

    if i == N_images: break


    



