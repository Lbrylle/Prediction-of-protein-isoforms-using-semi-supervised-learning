"""MAINSPLIT is a class for making the main data-split mentioned in the report. See the comments for more information.
"""


# imports
# TODO: REMOVE UNUSED IMPORTS
import h5py
#import re
import numpy as np
import torch.utils.data
from random import sample
#import time
import torch
from tqdm import tqdm
import pickle



class MainSplit(torch.utils.data.Dataset):
    def __init__(self, data_dir:str, train=True, normalize=True, N_unlabelled=167884):
        # number of unlabelled points
        self.N = N_unlabelled

        self.train = train
        f_archs4 = h5py.File(data_dir + 'archs4_gene_expression_norm_transposed.hdf5', mode='r')
        self.dset = f_archs4['expressions']
        
        f_gtex_gene = h5py.File(data_dir + 'gtex_gene_expression_norm_transposed.hdf5', mode='r')
        f_gtex_isoform = h5py.File(data_dir + 'gtex_isoform_expression_norm_transposed.hdf5', mode='r')


        self.dset_gene = f_gtex_gene['expressions']
        self.dset_isoform = f_gtex_isoform['expressions']

        tissue_types = np.array(f_gtex_gene['tissue'])
        unique_tissue_types = np.unique(tissue_types)

        idx = np.zeros(len(self.dset_gene))

        idx = []

        for tissue in unique_tissue_types:
            indexes = np.where(np.array(tissue_types) == tissue)[0]
            indexes = indexes.tolist()
            half_of_indexes = sample(indexes, int(len(indexes)//2))
            idx += half_of_indexes
        
        self.gtex_train_idx = idx
        self.gtex_test_idx = [i for i in range(len(self.dset_gene)) if i not in idx]

        if self.train:
            self.tissue_types = tissue_types[self.gtex_train_idx]
        else:
            self.tissue_types = tissue_types[self.gtex_test_idx]

        self.max = 23.66155 
        self.min = 0

        self.normalize = normalize

        if self.train: print(f"proportion of labelled in train: {(len(self.gtex_train_idx))/self.__len__():.2f}")


    def __len__(self):
        if self.train:
            return (len(self.gtex_train_idx) + len(self.dset)) - (len(self.dset) - self.N)
        else:
            return (len(self.gtex_test_idx))

    def __getitem__(self, idx):
        def normalize(data):
            if self.normalize:
                return (data - self.min)/(self.max - self.min)
            return data

        if self.train:
            if idx >= (len(self.gtex_train_idx)):
                idx_ = idx - len(self.gtex_train_idx)
                return normalize(self.dset[idx_]), np.zeros_like(self.dset_isoform[self.gtex_train_idx[0]])
            else:
                return normalize(self.dset_gene[self.gtex_train_idx[idx]]), self.dset_isoform[self.gtex_train_idx[idx]]

    
        else:
            return normalize(self.dset_gene[self.gtex_test_idx[idx]]), self.dset_isoform[self.gtex_test_idx[idx]]
               

def calculate_mean_sd(loader, N):
    """not so relevant, more of an experiment
    """
    x_sum = 0 #torch.zeros(18965).to(device)
    sum_sq = 0

    for (x, y) in tqdm(loader):
        x = x.to(device)
        x_sum = x_sum + x.sum()
        sum_sq = sum_sq + (x**2).sum()

    x_mean = x_sum/N

    var = (sum_sq - (x_sum**2)/N ) /(N-1)

    return x_sum/N, torch.sqrt(var)




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using ", device)


    from torch.utils.data import DataLoader
    from torch.utils.data import Subset
    from sklearn.model_selection import train_test_split
    batch_size = 64

    path_to_data = "/dtu-compute/datasets/iso_02456/hdf5-row-sorted/"

    train_split = MainSplit(path_to_data, train=True, N_unlabelled=167884)
    test_validation_split = MainSplit(path_to_data, train=False)


    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)

    
    test_idx, validation_idx = train_test_split(np.arange(len(test_validation_split)),
                                                test_size=0.4,
                                                random_state=999,
                                                shuffle=True,
                                                stratify=test_validation_split.tissue_types)


    test_dataset = Subset(test_validation_split, test_idx)
    validation_dataset = Subset(test_validation_split, validation_idx)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


    print(f"N train batches: {len(train_loader)}. N points: {len(train_split)}")
    print(f"N test batches: {len(test_loader)}. N points: {len(test_dataset)}")
    print(f"N validate batches: {len(validation_loader)}. N points: {len(validation_dataset)}")


    save = {}
    train_mean, train_sd = calculate_mean_sd(train_loader, len(train_split))
    save['train_mean'] = train_mean
    save['train_sd'] = train_sd
    
    print(train_mean, train_sd)

    with open('means.pickle', 'wb') as handle:
        pickle.dump(save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for (x, y) in train_loader:
        print(x, y)
        break
    
    for (x, y) in test_loader:
        print(x, y)
        break
    
    for (x, y) in validation_loader:
        print(x, y)
        break

    