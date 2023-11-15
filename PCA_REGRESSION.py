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


def init_dataloaders(batch_size, shuffle=True, seed = None):
        if seed:
            torch.manual_seed(seed)
        
        archs4_train = IsoDatasets.Archs4GeneExpressionDataset("/dtu-compute/datasets/iso_02456/hdf5/")
        archs4_train_dataloader = DataLoader(archs4_train, batch_size=batch_size, shuffle=shuffle)
        #
        gtex_train = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/", exclude='brain')
        gtex_test = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/" , include='brain')

        gtex_train_dataloader = DataLoader(gtex_train, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        gtex_test_dataloader = DataLoader(gtex_test, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        return archs4_train_dataloader, gtex_train_dataloader, gtex_test_dataloader


n_components = 100
batch_size = 100
n_epochs = 20

archs4_train_dataloader, gtex_train_dataloader, gtex_test_dataloader = init_dataloaders(batch_size=batch_size)

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

RegressionANN = nn.Sequential(
    nn.Linear(n_components, 1000),
    nn.ReLU(),
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 156958),
).to(device)


loss_fn = nn.MSELoss()
optimizer = optim.Adam(RegressionANN.parameters(), lr=0.001)


ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

# fit the ipca model on the archs4 data
print("Fitting ipca model on archs4 data...")
tic = time.time()
for X in archs4_train_dataloader:
    ipca.partial_fit(X.numpy())
    break

toc = time.time()
print(f"Time to fit ipca model on archs4 data: {toc-tic:.2f} seconds")


train_loss = np.zeros((n_epochs, len(gtex_train_dataloader)))
test_loss = np.zeros((n_epochs, len(gtex_test_dataloader)))


for epoch in range(n_epochs):
    RegressionANN.train()
    tic = time.time()
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
        if i == 2: break
    toc = time.time()
    print(f"Time to fit ANN on gtex data: {toc-tic:.2f} seconds")

    tic = time.time()

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
            if i == 2: break

    toc = time.time()
    print(f"Time to evaluate ANN on gtex data: {toc-tic:.2f} seconds")


# plotting
plt.figure(figsize=(10,10))
plt.plot(np.mean(train_loss, axis=1), label='train')
plt.plot(np.mean(test_loss, axis=1), label='test')
plt.legend()
plt.savefig('test_train.png')


print('train_loss for each epoch')
print(train_loss)

print('test_loss for each epoch')
print(test_loss)