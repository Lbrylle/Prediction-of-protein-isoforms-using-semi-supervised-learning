"""Plotting the saved error from the runs.
"""


# imports
import matplotlib.pyplot as plt
import numpy as np

# PLOT STYLE
#plt.style.use("fivethirtyeight")


# plot settings
if True: # dette er et eksperiment med nogle scienceplot-pakke
    import scienceplots
    plt.style.use('science')
    plt.rcParams.update({'figure.dpi': '200'})
    plt.rcParams.update({"legend.frameon": True})
    plt.rcParams.update({'font.size': 17})
    plt.rcParams.update({'font.weight': 'bold'})
    #plt.rcParams.update({'font.size': 30})
    #plt.rcParams.update({'font.weight': 'bold'})

# save location
save_path = '/zhome/51/4/167677/Desktop/PROJECT_DL/Final_run_1/local_plots/'




# PCA FFN ERROR
path_pca_fnntest = '/zhome/51/4/167677/Desktop/PROJECT_DL/results/19_12/Final_PCA_relu/pca_testfnn.npy'
path_pca_fnntrain = '/zhome/51/4/167677/Desktop/PROJECT_DL/results/19_12/Final_PCA_relu/pca_trainfnn.npy'

################################################################################

# M1 FFN ERROR
# AWAITING DATA
path_m1_fnntrain = '/zhome/51/4/167677/Desktop/PROJECT_DL/results/19_12/m1_final_run_relu_256_no_batchnorm/m1_trainfnn.npy'
path_m1_fnntest = '/zhome/51/4/167677/Desktop/PROJECT_DL/results/19_12/m1_final_run_relu_256_no_batchnorm/m1_testfnn.npy'


################################################################################

# FFN MSE ERROR
path_fnntrain = '/zhome/51/4/167677/Desktop/PROJECT_DL/results/05_12/FNN_FINAL/training_FNN.npy'
path_fnntest = '/zhome/51/4/167677/Desktop/PROJECT_DL/results/05_12/FNN_FINAL/validation_FNN.npy'

# THE MSE FROM USING THE MEAN OF Y's
y_mean_loss = np.load("/zhome/51/4/167677/Desktop/PROJECT_DL/results/19_12/y_mean/y_mean.npy")


# M2 DATA
import pickle
path_m2_fnntest = '/zhome/51/4/167677/Desktop/PROJECT_DL/results/19_12/m2/validation_data3.pkl'
path_m2_fnntrain = '/zhome/51/4/167677/Desktop/PROJECT_DL/results/19_12/m2/train_data3.pkl'


# Load the pickled object
with open(path_m2_fnntest, 'rb') as file:
    m2_fnntest = pickle.load(file)

with open(path_m2_fnntrain, 'rb') as file:
    m2_fnntrain = pickle.load(file)

#print(m2_fnntest)
#m2_loss = loaded_object['FNN']

pca_test_loss = np.load(path_pca_fnntest)
m1_test_loss = np.load(path_m1_fnntest) # awaiting data
fnn_test_loss = np.load(path_fnntest)
m2_test_loss = m2_fnntest['FNN']

pca_train_loss = np.load(path_pca_fnntrain)
m1_train_loss = np.load(path_m1_fnntrain) # awaiting data
fnn_train_loss = np.load(path_fnntrain)
m2_train_loss = m2_fnntrain['FNN']

#print(fnn_test_loss)
#plt.figure(figsize=(12,10))


#fig, axs = plt.subplots(1, 1, figsize=(6,6))
plt.figure(figsize=(12,6))
plt.tight_layout()
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.plot(np.arange(1,101), pca_test_loss, label='$\mathcal{M}_{PCA}$')
plt.plot(np.arange(1,101), m1_test_loss, label='$\mathcal{M}_{M1}$') # awaiting daata
plt.plot(np.arange(1,101), fnn_test_loss, label='$\mathcal{M}_{FNN}$')
plt.plot(np.arange(1,101), m2_test_loss, label='$\mathcal{M}_{M2}$')
plt.plot(np.arange(1,101), np.repeat(np.mean(y_mean_loss),100), '--')
plt.title('Test MSE')
plt.legend(loc='upper right')
plt.savefig(save_path + 'test_loss.png', bbox_inches="tight")
plt.close()


#fig, axs = plt.subplots(1, 1, figsize=(6,6))

plt.figure(figsize=(12,6))
plt.tight_layout()
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.plot(np.arange(1,101), pca_train_loss, label='$\mathcal{M}_{PCA}$')
plt.plot(np.arange(1,101), m1_train_loss, label='$\mathcal{M}_{M1}$') # awaiting data
plt.plot(np.arange(1,101), fnn_train_loss, label='$\mathcal{M}_{FNN}$')
plt.plot(np.arange(1,101), m2_train_loss, label='$\mathcal{M}_{M2}$')
plt.title('Train MSE')
plt.legend(loc = 'upper right')
plt.savefig(save_path + 'train_loss.png', bbox_inches="tight")
plt.close()


print("PCA: mean error of last epochs:", np.mean(pca_test_loss[-4:]))
print("M1: mean error of last epoch:", np.mean(m1_test_loss[-4:]))
print("FNN: mean error of last epochs:", np.mean(fnn_test_loss[-4:]))
print("y_mean:",  np.mean(y_mean_loss))

