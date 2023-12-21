"""Feature extraction with Iterative PCA and training on the extracted features with FNN

This works as a baseline.
"""

# imports 
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import time 
from collections import defaultdict
import numpy as np
import pickle
from modules.helperFunctions import *

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")




from modules.MainSplit import MainSplit
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

batch_size = 128
n_components = 128
n_epochs = 100
experiment_name = 'FINAL_FINAL_PCA'
output_dim = 156958
learning_rate = 1e-4
random_seed_ = 1


path_to_data = "/dtu-compute/datasets/iso_02456/hdf5-row-sorted/"

save_path = init_folders(experiment_name)

random_seed(random_seed_)

save_parameters(save_path,
                batch_size=batch_size,
                n_components=n_components,
                n_epochs=n_epochs,
                experiment_name=experiment_name,
                output_dim=output_dim,
                lr=learning_rate,
                random_seed_=random_seed_
                )

# IPCA will error if this is not uphold
assert n_components <= batch_size


# THE MAIN TRAINING SPLIT (some y's are 'labelled')
train_split = MainSplit(path_to_data, train=True)
train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, drop_last=True)



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




#* Build the FFN
regressor = nn.Sequential(
            nn.Linear(n_components, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 156958)  # Output dimension for regression
        ).to(device)



# define the loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(regressor.parameters(), lr=learning_rate)

# initialize the ipca model
ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

# save time
print("Fitting ipca model on train data...")
tic = time.time()

# fit the ipca model on the archs4 data
def train_pca(dataloader, model):
    for (x, y) in tqdm(dataloader, desc='PCA - Training'):
        model.partial_fit(x)

from joblib import dump, load

train_pca(train_loader, ipca)

dump(ipca, save_path + '/ipca.joblib') 


# print time
toc = time.time()
print(f"Time to fit ipca model on train data: {toc-tic:.2f} seconds")



def plot_temp(lists:dict, save_path):
    plt.figure()
    for n, v in lists.items():
        plt.plot(v, label='n')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def train_fnn(ipca,
              optimizer, 
              regressor, 
              train_data_loader,
              test_data_loader,
              num_epochs, 
              device,
              criterion):
    epoch = 0
    regressor.train()
    save_data = defaultdict(list)
    with tqdm(total=num_epochs * len(train_data_loader) * len(test_data_loader), desc='FNN - Training') as pbar:
        while epoch < num_epochs:
            epoch += 1
            epoch_data = defaultdict(list)
            
            for i, (x, y) in enumerate(train_data_loader):
                latent = ipca.transform(x)
                latent = torch.from_numpy(latent).to(device).float()

                y = y.to(device)

                y_pred = regressor(latent)
                loss = criterion(y_pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_data['fnn_trainloss'].append(loss.item())

                pbar.update(1)

                if i % 100:
                    tqdm.write("FNN_trainloss: " + str(loss.item()))


            
                
            for i, (x,y) in enumerate(test_data_loader):
                latent = ipca.transform(x)
                latent = torch.from_numpy(latent).to(device).float()

                y = y.to(device)

                y_pred = regressor(latent)
                loss = criterion(y_pred, y)
                optimizer.step()

                epoch_data['fnn_testloss'].append(loss.item())

                pbar.update(1)

                if i % 100:
                    tqdm.write("FNN_testloss: " + str(loss.item()))
       
            



            #plot_temp({'fnn_trainloss': epoch_data['fnn_trainloss']}, save_path + '/pca_fnn_trainloss' + str(epoch)+ '.png')
            #plot_temp({'fnn_testloss': epoch_data['fnn_testloss']}, save_path + '/pca_fnn_testloss' + str(epoch)+ '.png')
            save_data['fnn_trainloss'].append(np.mean(epoch_data['fnn_trainloss']))
            save_data['fnn_testloss'].append(np.mean(epoch_data['fnn_testloss']))


            np.save(save_path + 'pca_trainfnn.npy', save_data['fnn_trainloss'])
            np.save(save_path + 'pca_testfnn.npy', save_data['fnn_testloss'])

            np.save(save_path + 'fnntest_last_epoch.npy', epoch_data['fnn_testloss'])

            torch.save(regressor.state_dict(), save_path + '/regressor.pth')





criterion = nn.MSELoss()

save_fnn = train_fnn(ipca, 
                    optimizer, 
                    regressor, 
                    test_loader,
                    validation_loader, 
                    n_epochs, 
                    device, 
                    criterion)








