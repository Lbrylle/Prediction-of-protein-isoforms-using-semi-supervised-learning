from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import modules.IsoDatasets as IsoDatasets
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
""" Creates a tSNE plot of the GTEx dataset """

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load the GTEx dataset
gtex_train = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/", exclude='brain')
gtex_test = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/" , include='brain')

# Create a dataloader for the GTEx dataset
gtx_train_dataloader = DataLoader(gtex_train, batch_size=156958, shuffle=True)

# Iterate over the dataloader and get the X, y and tissue
for X, y, tissue in tqdm(gtx_train_dataloader):
    # Move data to GPU if available
    X = X.to(device)
    y = y.to(device)

    # Move data back to CPU before using TSNE
    X_cpu = X.cpu().detach().numpy()
    y_cpu = y.cpu().detach().numpy()
        
    # We want to get TSNE embedding with 2 dimensions
    n_components = 2
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(X_cpu)

    # Plot the result of our TSNE with the label color coded
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': tissue})
    fig, ax = plt.subplots(1, figsize=(10,10))  # Set the initial figure size here
    plt.figure(figsize=(16, 16))  # Set the figure size before plotting
    plt.style.use("fivethirtyeight")

    # Plot the result of our TSNE with the label color coded
    sns.scatterplot(x='tsne_1', y='tsne_2',  hue='label', data=tsne_result_df, ax=ax, s=120)

    # Set the limits of the plot
    lim = (tsne_result.min() - 1, tsne_result.max() + 1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')

    # Add a legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    # Save the plot
    fig.savefig("plots/tSNE.png", bbox_inches='tight')

    break