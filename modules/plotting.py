import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def createLossPlotFNN(input_train, input_val, name):
    plt.figure(figsize=(10, 5))
    plt.title(name)

    plt.plot(input_train, label='Training')
    plt.plot(input_val, label='Validation')
    plt.legend()
    plt.tight_layout()
    tmp_img = 'plots/' + name + '.png'
    plt.savefig(tmp_img)

def createLossPlotVAE(training_loss, validation_loss, epoch):
    #print("T-loss", training_loss)
    #print("V-loss", validation_loss)
    plt.figure(figsize=(10, 5))
    plt.title(r'Loss')
    plt.plot(training_loss, label='Training')
    plt.plot(validation_loss, label='Validation')
    plt.legend()
    plt.tight_layout()
    tmp_img = 'plots/loss.png'
    plt.savefig(tmp_img)

def createELBOPlotVAE(training_data, validation_data, epoch):

    plt.figure(figsize=(10, 5))
    plt.title(r'ELBO: $\mathcal{L} ( \mathbf{x} )$')
    plt.plot(training_data['elbo'], label='Training')
    plt.plot(validation_data['elbo'], label='Validation')
    plt.legend()
    plt.tight_layout()
    tmp_img = 'plots/elbo.png'
    plt.savefig(tmp_img)

    