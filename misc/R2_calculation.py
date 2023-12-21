"""Calculation of the Coefficient of Determination

This is solely based on the r2_score by sklearn. This script calculates the Coefficient of Determination R^2 for the following models:
    FNN trained on PCA features
    FNN trained on M1 features
    FNN trained directly on data. 
    (maybe include regressor from M2)


OBS: THIS NEEDS TO BE MOVED ONE DIRECTORY IF YOU WISH TO RUN IT
"""



# imports. 
from torch import nn, Tensor
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from torch.distributions import Normal
from modules.helperFunctions import ReparameterizedDiagonalGaussian, random_seed
import random


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

    return train_loader, test_loader, validation_loader




# parameters
latent_features = 512
output_dim = 156958
input_dim = 18965
batch_size = 2
random_seed_ = 1
n_components = 512

path_to_data = "/dtu-compute/datasets/iso_02456/hdf5-row-sorted/"

random_seed(random_seed_)



############################################################################################################
#* LOAD PCA BASED FNN & IPCA SAVE
############################################################################################################

regressor_pca = nn.Sequential(
            nn.Linear(n_components, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 156958)  # Output dimension for regression
        ).to(device)



pca_pth = '/zhome/51/4/167677/Desktop/PROJECT_DL/results/19_12/Final_PCA_relu/regressor.pth'

#* LOADING THE TRAINED FNN
regressor_pca.load_state_dict(torch.load(pca_pth))
regressor_pca.eval()


#* LOADING THE TRAINED IPCA
from joblib import dump, load

path_joblib = '/zhome/51/4/167677/Desktop/PROJECT_DL/results/19_12/Final_PCA_relu/ipca.joblib'
icpa = load(path_joblib) 


############################################################################################################
#* LOAD FNN TRAINED DIRECTLY ON DATA
############################################################################################################

class FNN(nn.Module):
    def __init__(self, input_dim:int, N_isoform:int, p=0.5):
        
        super(FNN, self).__init__()

        # FNN
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Dropout(p=p),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(p=p),
            nn.ReLU(),
            nn.Linear(1024, 2*N_isoform)
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.regressor.apply(init_weights)


    def forward(self, x: Tensor) -> Tensor:
        mu, log_var = self.regressor(x).chunk(2, dim=-1)
        #print(mu.shape, log_var.shape)
        p_y = Normal(mu, log_var.exp())
        return p_y.rsample()


regressor_ffn = FNN(input_dim, output_dim).to(device)


ffn_pth = '/zhome/51/4/167677/Desktop/PROJECT_DL/results/19_12/FNN_final_relu/FNN.pth'

regressor_ffn.load_state_dict(torch.load(ffn_pth))
regressor_ffn.eval()

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


m1_ffn_pth = '/zhome/51/4/167677/Desktop/PROJECT_DL/results/19_12/m1_final_run_relu_256_no_batchnorm/regressor.pth'


regressor_m1 = nn.Sequential(
                nn.Linear(latent_features, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, output_dim)  # Output dimension for regression
            ).to(device)


regressor_m1.load_state_dict(torch.load(m1_ffn_pth))
regressor_m1.eval()



############################################################################################################
#* LOAD M2 FFN
############################################################################################################


regressor_m2 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),  
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2*output_dim)
        )

m2_ffn_pth = 'path_to_m2_ffn'

regressor_m2.load_state_dict(torch.load(m2_ffn_pth))
regressor_m2.eval()



############################################################################################################
#* LOAD DATA
############################################################################################################

#path_to_data = "/dtu-compute/datasets/iso_02456/hdf5/"

train_loader, test_loader, validation_loader = init_dataloaders(batch_size)


############################################################################################################
#* CALCULATING THE MEAN
############################################################################################################


y_save = torch.zeros(1, output_dim)


N = len(validation_loader)

for (_,y) in tqdm(validation_loader, desc='y_mean'):
     a = y.sum(0).reshape(1, -1)
     y_save = y_save + a


y_mean = y_save/N


def y_mean_pred(batch_size):
    return np.repeat(y_mean.numpy(), batch_size, axis=0)


############################################################################################################
#* CALCULATE R^2
############################################################################################################

# helper functions
def to_numpy(tensor_):
    #print(tensor_)
    return tensor_.detach().cpu().numpy()

def to_tensor(array):
    return torch.from_numpy(array).float().to(device)




pca_RSS, m1_RSS, m2_RSS, fnn_RSS, mean_RSS = 0, 0, 0, 0, 0

SST = 0

for i, (x,y) in tqdm(enumerate(test_loader), desc='R2'):
    # x is tensor
    # y is tensor
    
    #y = to_numpy(y)
    y = y.to(device)

    etot = y - to_tensor(y_mean_pred(batch_size))
    SST += (etot @ etot.T).sum().item()
 #   SST += torch.dot(etot, etot).sum().item()


    x = x.to(device)
    #* PCA
    latent = icpa.transform(to_numpy(x))

    y_pred = regressor_pca(to_tensor(latent))

    e = y - y_pred
    pca_RSS += torch.sum(e @ e.T).item()

    #print(np.corrcoef(to_numpy(y), to_numpy(y_pred)))

    #* FNN
    y_pred = regressor_ffn(x)

    e = y - y_pred

    fnn_RSS += torch.sum(e @ e.T).item()



    #* M1
    latent = encode(x)

    y_pred = regressor_m1(latent)

    e = y - y_pred

    m1_RSS += torch.sum(e @ e.T).item()

    #* M2
    y_pred = regressor_m2(x)

    e = y - y_pred

    m2_RSS += torch.sum(e @ e.T).item()


    #* mean
    #mean_r2 += np.sum(r2_score(y, y_mean_pred(batch_size), multioutput='raw_values'))

    if i%10:
        print('\n' + '-'*20 + '\n')
        print(i)
        print('\n')
        print('pca', 1 - pca_RSS/SST)
        print('fnn', 1 - fnn_RSS/SST)
        print('m1_r2', 1 - m1_RSS/SST)
        print('m2_r2', 1 - m2_RSS/SST)
        #print('mean_r2', mean_r2)


print('\n'*3 + '-'*20 + '\n')
print('FINAL VALUES:')
print('pca', 1 - pca_RSS/SST)
print('fnn', 1 - fnn_RSS/SST)
print('m1_r2', 1 - m1_RSS/SST)
print('m2_r2', 1 - m2_RSS/SST)
#print('mean_r2', mean_r2/N)