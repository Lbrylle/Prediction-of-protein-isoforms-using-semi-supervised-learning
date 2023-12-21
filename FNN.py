"""A Feed-Forward Neural-Network (FNN) for predicting isoform expression directly from gene expression.

This script is used to train a FNN on the labelled part of the main split of the data.
"""


# imports
from torch import nn, Tensor
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
from modules.plotting import *
from modules.helperFunctions import *
from torch.distributions import Normal



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)



############################################################################################################
#* THE MAIN MODEL (FNN)
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
        print(mu.shape, log_var.shape)
        p_y = Normal(mu, log_var.exp())
        return p_y.rsample()

    

if __name__ == '__main__':
    
    # Saving parameters of the run
    path_to_data = "/dtu-compute/datasets/iso_02456/hdf5-row-sorted/"
    input_dim = 18965
    N_isoform = 156958
    num_epochs = 100
    random_seed_ = 1
    batch_size = 64
    learning_rate = 1e-4
    experiment_name = 'FNN_final_relu'


    # initialize folder structure
    save_path = init_folders(experiment_name)


    # save parameters of current run
    save_parameters(save_path, 
                    experiment_name=experiment_name,
                    input_dim=input_dim, 
                    N_isoform=N_isoform, 
                    num_epochs=num_epochs, 
                    random_seed_=random_seed_, 
                    batch_size=batch_size,
                    learning_rate=learning_rate)

    # random seeding
    random_seed(random_seed_)



    ############################################################################################################
    #* THE DATA (MainSplit)
    ############################################################################################################


    # THE MAIN TRAINING SPLIT
    from torch.utils.data import DataLoader
    from modules.MainSplit import MainSplit
    from torch.utils.data import Subset
    from sklearn.model_selection import train_test_split


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


    train_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True)




    ############################################################################################################
    #* TRAINING THE MODEL
    ############################################################################################################


    model = FNN(input_dim, N_isoform)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=1e-4)

    # Training params



    def train(model, optimizer, num_epochs, criterion = nn.MSELoss()):
        epoch = 0
        train_data = defaultdict(list)
        validation_data = defaultdict(list)

        while epoch < num_epochs:
            epoch += 1
            training_epoch_data = defaultdict(list)
            validation_epoch_data = defaultdict(list)

            model.train()
            for (x, y) in tqdm(train_loader, desc = "FNN - Training"):
                y = y.to(device)
                x = x.to(device)

                optimizer.zero_grad()

                y_pred = model(x)

                loss = criterion(y_pred, y)
                print(loss)
                training_epoch_data["FNN"].append(loss.item())

                loss.backward()

                clipping_value = 0.5 # arbitrary value of your choosing
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)

                optimizer.step()
                
                training_epoch_data["loss"].append(loss.item())
                

            train_FNN = np.mean(training_epoch_data["FNN"])
            train_data["FNN"].append(train_FNN)

            with torch.no_grad():
                model.eval()

                for x, y in tqdm(validation_loader, desc = "FNN - Validation"):
                    y = y.to(device)
                    x = x.to(device)

                    y_pred = model(x)

                    loss = criterion(y_pred, y)

                    validation_epoch_data["FNN"].append(loss.item())


                validation_FNN = np.mean(validation_epoch_data["FNN"])
                validation_data["FNN"].append(validation_FNN)


            # save data
            np.save(save_path + '/training_FNN.npy', train_data['FNN'])
            np.save(save_path + '/validation_FNN.npy', validation_data['FNN'])
            np.save(save_path + '/last_epoch_FNN.npy', validation_epoch_data["FNN"])

            torch.save(model.state_dict(), save_path + '/FNN.pth')



    train(model, optimizer, num_epochs)        






