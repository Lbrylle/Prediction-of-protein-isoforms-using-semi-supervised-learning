# 02456 - Project: Prediction-of-protein-isoforms-using-semi-supervised-learning

This is the github-repository associated with project *Prediction of protein isoforms using semi-supervised learning* by Lucas Brylle (s203832) and Nikolaj Hertz (s214644) of the Technical University of Denmark. 

## Files

* AllModels.ipynb: *THE REQUIRED JUPYTER NOTEBOOK FOR RECREATING RESULTS*

* M1.py: The main file for the M1 (unsupervised VAE)

* M2.py: The main file for the M2 (semi-supervised VAE)

* PCA.py: The main file for the PCA framework

* FNN.py: The main file for the FNN (Feed-forward neural network)

*In the misc folder one would find:*
* misc/distribution_of_y.py: script for creating the appendix A

* misc/distribution_of_z.py: script for creating the appendix B

* misc/plot_elbo: script for creating the plots associated with the loss of M1 and M2

* misc/plot_mse: script for creating the plots associated with the MSE loss of all the models

* misc/R2_calculation: script for calculating the **Coefficient of Determination** ($R^2$-value). 

In the modules folder one would find:

* modules/helperFunctions.py: file containing some helper functions for running the large models

* modules/IsoDatasets.py: file containing the pytorch dataset given by the supervisors

* modules/MainSplit.py: modified version of the pytorch datasets contained in the IsoDatasets, in order to create the stratified main split of the data

* modules/plotting.py: some plotting functions (used for showing performance under training)


