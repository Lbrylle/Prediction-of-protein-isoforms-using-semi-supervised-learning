# imports 
import IsoDatasets
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import time 
import numpy as np


#* TODO: do cross-validation?


# helper functions
def save_parameters(save_path, **args):
    with open(save_path + 'parameters.txt', 'w') as f:
        for key, value in args.items():
            f.write(f"{key}: {value}\n")
    

def init_folders(experiment_name):
    today = time.strftime("%d_%m")
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('results/' + today):
        os.mkdir('results/' + today)
    if not os.path.exists('results/' + today + '/' + experiment_name):
        os.mkdir('results/' + today + '/' + experiment_name)
    save_path = 'results/' + today + '/' + experiment_name + '/'


    return save_path

# function to initialize dataloaders
def init_dataloaders(batch_size, shuffle=True, seed = None, drop_last=True):
        if seed:
            torch.manual_seed(seed)
        
        archs4_train = IsoDatasets.Archs4GeneExpressionDataset("/dtu-compute/datasets/iso_02456/hdf5/")
        archs4_train_dataloader = DataLoader(archs4_train, batch_size=batch_size, shuffle=shuffle)
        #
        gtex_train = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/", exclude='brain')
        gtex_test = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/" , include='brain')

        gtex_train_dataloader = DataLoader(gtex_train, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        gtex_test_dataloader = DataLoader(gtex_test, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        return archs4_train_dataloader, gtex_train_dataloader, gtex_test_dataloader


# parameters to change
n_components = 100
batch_size = 100
assert n_components <= batch_size
n_epochs = 20
experiment_name = 'test_run1'
hidden_size = [1000, 1000]
output_dim = 156958
learning_rate = 0.001




save_path = init_folders(experiment_name)

archs4_train_dataloader, gtex_train_dataloader, gtex_test_dataloader = init_dataloaders(batch_size=batch_size)

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


# save parameters in txt file
save_parameters(save_path, 
                n_components=n_components,
                batch_size=batch_size,
                n_epochs=n_epochs,
                experiment_name=experiment_name,
                hidden_size=hidden_size,
                output_dim=output_dim,
                learning_rate=learning_rate,
                device=device)



#* Build the ANN

modules = []

# add the hidden layers
for i, hidden in enumerate(hidden_size):
    if i == 0:
        modules.append(nn.Linear(n_components, hidden_size[i]))
        modules.append(nn.ReLU())
    else:
        modules.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
        modules.append(nn.ReLU())

modules.append(nn.Linear(hidden_size[-1], output_dim))
modules.append(nn.ReLU())

RegressionANN = nn.Sequential(*modules).to(device)




# define the loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(RegressionANN.parameters(), lr=learning_rate)

# initialize the ipca model
ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

# save time
print("Fitting ipca model on archs4 data...")
tic = time.time()

# fit the ipca model on the archs4 data
for X in archs4_train_dataloader:
    ipca.partial_fit(X.numpy())


# print time
toc = time.time()
print(f"Time to fit ipca model on archs4 data: {toc-tic:.2f} seconds")


# initilize arrays to store the train and test loss
train_loss = np.zeros((n_epochs, len(gtex_train_dataloader)))
test_loss = np.zeros((n_epochs, len(gtex_test_dataloader)))


# main training loop
for epoch in range(n_epochs):
    tic = time.time()

    # train the ANN on the gtex train data
    RegressionANN.train()
    for i, (X,y) in enumerate(gtex_train_dataloader):
        X = ipca.transform(X)
        X = torch.from_numpy(X).float()
        X = X.to(device)

        y = y.to(device)

        y_pred = RegressionANN(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss[epoch, i] = loss.item()

    toc = time.time()
    print(f"Time to fit ANN on gtex data: {toc-tic:.2f} seconds")


    tic = time.time()
    # evaluate the ANN on the gtex test data
    RegressionANN.eval()
    with torch.no_grad():
        for i, (X,y) in enumerate(gtex_test_dataloader):
            X = ipca.transform(X)
            X = torch.from_numpy(X).float()
            X = X.to(device)

            y = y.to(device)

            y_pred = RegressionANN(X)
            loss = loss_fn(y_pred, y)

            test_loss[epoch, i] = loss.item()

    toc = time.time()
    print(f"Time to evaluate ANN on gtex data: {toc-tic:.2f} seconds")


# plotting
plt.figure(figsize=(10,10))
plt.plot(np.mean(train_loss, axis=1), label='train')
plt.plot(np.mean(test_loss, axis=1), label='test')
plt.legend()
plt.savefig(save_path + 'loss_plot.png')

# save the arrays as csv
np.savetxt(save_path + "train_loss.csv", train_loss, delimiter=",")
np.savetxt(save_path + "test_loss.csv", test_loss, delimiter=",")

