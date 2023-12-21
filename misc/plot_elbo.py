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
    plt.style.use(['science','ieee'])
    plt.rcParams.update({'figure.dpi': '200'})
    plt.rcParams.update({"legend.frameon": True})
    plt.rcParams.update({'font.size': 17})
    plt.rcParams.update({'font.weight': 'bold'})



# save location
save_path = '/zhome/51/4/167677/Desktop/PROJECT_DL/Final_run_1/local_plots/'



path_m1_testloss = '/zhome/51/4/167677/Desktop/PROJECT_DL/results/19_12/m1_final_run_relu_256_no_batchnorm/m1_testloss.npy'
path_m1_trainloss = '/zhome/51/4/167677/Desktop/PROJECT_DL/results/19_12/m1_final_run_relu_256_no_batchnorm/m1_trainloss.npy'


import pickle
path_m2_fnntest = '/zhome/51/4/167677/Desktop/PROJECT_DL/results/19_12/m2/validation_data3.pkl'
path_m2_fnntrain = '/zhome/51/4/167677/Desktop/PROJECT_DL/results/19_12/m2/train_data3.pkl'


# Load the pickled object
with open(path_m2_fnntest, 'rb') as file:
    m2_fnntest = pickle.load(file)

with open(path_m2_fnntrain, 'rb') as file:
    m2_fnntrain = pickle.load(file)



m2_vae_train = m2_fnntrain['loss']
m2_vae_test = m2_fnntest['loss']

#m2_loss = loaded_object['FNN']

m1_testloss = np.load('/zhome/51/4/167677/Desktop/PROJECT_DL/results/19_12/m1_final_run_relu_256_no_batchnorm/m1_testloss.npy')
m1_trainloss = np.load('/zhome/51/4/167677/Desktop/PROJECT_DL/results/19_12/m1_final_run_relu_256_no_batchnorm/m1_trainloss.npy')

#print(fnn_test_loss)
#plt.figure(figsize=(12,10))


plt.figure(figsize=(6,6))
plt.tight_layout()
plt.ylabel('$\mathcal{J}$ (negative elbo)')
plt.xlabel('Epoch')
plt.plot(np.arange(1,len(m1_testloss) + 1), m1_testloss, label='Train loss')
plt.plot(np.arange(1,len(m1_trainloss) + 1), m1_trainloss, label='Test loss')
plt.title('$\mathcal{M}_{M1}$')
plt.legend()
plt.savefig(save_path + 'm1_loss.png', bbox_inches="tight")
plt.close()

plt.figure(figsize=(6,6))
plt.tight_layout()
plt.ylabel('$\mathcal{J}^{\\alpha}$ (negative elbo)')
plt.xlabel('Epoch')
plt.plot(np.arange(0,100), m2_vae_train, label='Train loss')#, label='$E_{M2}$')
plt.plot(np.arange(0,100), m2_vae_test, label = 'Test loss')#, label='$E_{M2}$')
plt.title('$\mathcal{M}_{M2}$')
plt.legend()
plt.savefig(save_path + 'm2_loss.png', bbox_inches="tight")
plt.close()


