# 02456 - Project: Prediction-of-protein-isoforms-using-semi-supervised-learning
This is the github-repository associated with project *Prediction of protein isoforms using semi-supervised learning* by Lucas Brylle (s203832) and Nikolaj Hertz (s214644) from the Technical University of Denmark. It explores the application of semi-supervised learning to predict protein isoforms.

## Abstract
Protein isoforms are generally understudied compared to gene expressions and to address this gap, this project seeks to build a framework based on Variational Autoencoders to predict the isoform expression. Specifically, a comparison between an unsupervised and a semi-supervised approach will be performed, where it is examined whether including some of the labels hereby constructing a semi-labelled training data split, will improve the learning process of the model. The models are compared to benchmarks models such as a regressor trained on the extracted features from a simple PCA. The described methods result in a sub-optimal performance for the semi-supervised setup, compared to the other methods. However, this can be due to some of the chosen distributions, the tuning of the hyper-parameters and the basic notion of how applicable the model is in the setting. 

## Trained models
Due to the size of the trained models, they can be found and dowloaded through the following google drive:
https://drive.google.com/drive/folders/1vJQ_KLSOcMv2daaFpuJoyUxyqpcWQDX5?usp=drive_link


## Files

* AllModels.ipynb: *THE REQUIRED JUPYTER NOTEBOOK FOR RECREATING RESULTS*

* M1.py: The main file for the M1 (unsupervised VAE)

* M2.py: The main file for the M2 (semi-supervised VAE)

* PCA.py: The main file for the PCA framework

* FNN.py: The main file for the FNN (Feed-forward neural network)

* tSNE.py: The main file for creating figure 1. 

*In the misc folder one would find:*
* misc/distribution_of_y.py: script for creating the appendix A

* misc/distribution_of_z.py: script for creating the appendix B

* misc/plot_elbo.py: script for creating the plots (figure 3) associated with the loss of M1 and M2

* misc/plot_mse.py: script for creating the plots (figure 4) associated with the MSE loss of all the models

* misc/R2_calculation.py: script for calculating the **Coefficient of Determination** ($R^2$-value). 

* misc/GaussianIsoforms.csv: the determined means and standard deviations of the $y$'s

*In the modules folder one would find:*

* modules/helperFunctions.py: file containing some helper functions for running the large models

* modules/IsoDatasets.py: file containing the pytorch dataset given by the supervisors

* modules/MainSplit.py: modified version of the pytorch datasets contained in the IsoDatasets, in order to create the stratified main split of the data

* modules/plotting.py: some plotting functions (used for showing performance under training)

*In the epoch_data directory one would find:*

* epoch_data/M2_train.pkl: the train ELBO and train MSE error for $\mathcal{M}_{M2}$

* epoch_data/M2_test.pkl: the test ELBO and test MSE error for $\mathcal{M}_{M2}$

* epoch_data/M1_*loss.npy: two files containing the test and train loss for the ELBO (equation 1)

* epoch_data/m1_*fnn.npy: two files containing the test and train MSE for the regressor.

* epoch_data/pca_train.npy: the train MSE error for regressor trained on PCA features

* epoch_data/pca_test.npy: the test MSE error for regressor trained on PCA features

* epoch_data/fnn_train.npy: the train error for the Feed-forward Neural Network

* epoch_data/fnn_test.npy: the test error for the Feed-forward Neural Network

* epoch_data/y_mean.npy: the mse for every datapoint using the mean of the y's. 



## Contributions
For contributions or queries, please contact Lucas Brylle (s203832@student.dtu.dk) or Nikolaj Hertz (s214644@student.dtu.dk)

## Acknowledgements
Special thanks to our supervisors Jes Frellsen (jefr@dtu.dk) and Kristoffer Vitting-Seerup (krivi@dtu.dk)


