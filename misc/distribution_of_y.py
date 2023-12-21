"""local test of y distribution

picking a isoform and checking if it normal distributed
"""

from sklearn.metrics import r2_score
from torch import nn, Tensor
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from modules.helperFunctions import ReparameterizedDiagonalGaussian, random_seed, init_folders
import matplotlib.pyplot as plt
import os
import time
import random
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from modules.MainSplit import MainSplit
import pandas as pd

# plot settings
import scienceplots
plt.style.use('science')
plt.rcParams.update({'figure.dpi': '200'})
plt.rcParams.update({"legend.frameon": True})




# parameters
latent_features = 512
output_dim = 156958
input_dim = 18965
batch_size = 1
random_seed_ = 1

experiment_name = 'y_plots'

save_path = init_folders(experiment_name)

path_to_data = "/dtu-compute/datasets/iso_02456/hdf5-row-sorted/"

random_seed(random_seed_)



from modules.IsoDatasets import GtexDataset



gtex = GtexDataset(path_to_data)

data_loader = DataLoader(gtex, batch_size=batch_size, shuffle=True)




def gauss_pdf(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi) * sigma) * np.exp(-1/(2*sigma**2)*(x - mu)**2)


old_path = '/zhome/51/4/167677/Desktop/PROJECT_DL/Final_run_1/GaussianIsoforms.csv'
py_csv = pd.read_csv(old_path)


dim = 0

dims = [0, 10, 100, 1000]

for dim in dims:
    print(f"dimension {dim}")
    mean = py_csv['Mean'][dim]
    sd = py_csv['Standard_Deviation'][dim]


    list_of_ys = np.zeros(17356)

    for i, (x,y) in tqdm(enumerate(data_loader)):
        #print(y.shape)
        list_of_ys[i] = y[0,dim]





    xmin = mean - 5*sd
    xmax = mean + 5*sd

    xs = np.linspace(xmin, xmax, 100)

    y = gauss_pdf(xs, mean, sd)

    fig, axs = plt.subplots(1, 2, figsize=(7,4))
    plt.tight_layout()
    axs[0].set_xlabel('$\mathbf{y}$' + f' (dimension {dim})')
    axs[0].set_ylabel('Frequency')
    axs[0].plot(xs, y, 'r--')
    axs[0].hist(list_of_ys, bins=200, density=True)



    index = (list_of_ys == 0)

    axs[1].set_xlabel('$\mathbf{y}$' +  f' (dimension {dim})')
    axs[1].set_ylabel('Frequency')
    axs[1].plot(xs, y, 'r--')
    axs[1].hist(list_of_ys[~index], bins=200, density=True)
    axs[1].set_title('Without zeros')


    tmp_img = save_path + f'test_y_{dim}.png'
    plt.savefig(tmp_img)
    plt.close()









