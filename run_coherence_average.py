# %%
# import the main libraries required for the preprocessing
from mne_bids import BIDSPath
import mne 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne_connectivity
import os 
import pickle
import pandas as pd 
# import homemade function 
from technical_validation_utils import (parula_map, save_multipage)
from scipy import interpolate

mne.set_log_level(verbose='CRITICAL') # reduce verbose output
# %matplotlib qt
    
# %%
# create a list of subjects to loop over all the fif file in session PeriOp
# remove empty room and split 02 as it is read when we read split 01
bids_root = '/data/raw/hirsch/RestHoldMove_anon/'
subjects = [sub for sub in os.listdir(bids_root) if os.path.isdir(bids_root + sub)]
subjects.sort() # get all subjects
coherence_dir = [Path('./coherence/') / sub for sub in subjects]

# %%

# store variable for coherences 
all_coherences = []
all_coherences_left = []
all_coherences_right = []
got_info = False # check if we stored the MEG info with 204 sensors

for n_sub, subject_dir in enumerate(coherence_dir):
    coh_files = [subject_dir / file for file in os.listdir(subject_dir) if 'coh' in file]
    info_coh_files = [subject_dir / file for file in os.listdir(subject_dir) if 'ind' in file]
    # to make them match
    coh_files.sort() 
    info_coh_files.sort()

    # store all coherences for one subjects to average over
    coherences = []
    coherences_left = []
    coherences_right = []
    
    # loop trough each coherence file
    for coh_file, info_file in zip(coh_files, info_coh_files):
        # read the file
        coh = mne_connectivity.read_connectivity(coh_file)
        with open(info_file, 'rb') as f:
            coh_info = pickle.load(f)
            
        # get the data of the connectivity coherence --> shape should be (N_connection * N_freqs)
        coh_data = coh.get_data() 
        coh_freqs = np.array(coh.freqs)
        lfp_left_ind = coh_info['lfp_left_ind']
        lfp_right_ind = coh_info['lfp_right_ind']
        meg_info = coh_info['meg_info']
        
        # sometime 2 bad MEG channels were removed, so we get the sensors position for this 2 missing sensors
        if len(meg_info.ch_names) == 204 and got_info != True:
            meg_info_max = meg_info
            got_info = False
    
        # reshape the data according to (N_lfp * N_meg * N_freqs)
        coh_data = coh_data.reshape(coh_info['n_lfp_sensors'], coh_info['n_meg_sensors'], len(coh_freqs))    
        
        # average the connectivity on LFP axis channel, average type: either all LFP, only left or right side
        coh_average = np.mean(coh_data, axis=0)

        # assert that the lfp_side_ind is not empty and average over the side
        if len(lfp_left_ind) > 0:
            coh_average_left = np.mean(coh_data[lfp_left_ind, :, :], axis=0)
            
        if len(lfp_right_ind) > 0:
            coh_average_right = np.mean(coh_data[lfp_right_ind, :, :], axis=0)

        # append the coherence for each file of the subject
        coherences.append(coh_average)
        coherences_left.append(coh_average_left)
        coherences_right.append(coh_average_right)

    # average on files axis and append it to the grand average list
    all_coherences.append(np.mean(coherences, axis=0))
    all_coherences_left.append(np.mean(coherences_left, axis=0))
    all_coherences_right.append(np.mean(coherences_right, axis=0))

 
# %%
# create a function to pad the grand average.sometime the shape of MEG sensors is 204 or 202.
# here we reshape the sensors column to the maximum size of the array so that every coherence have 204 sensors.
# we fill the extra array with nan when it does not already exist. (i.e. if 202 it will be 204 and fill the extra with nan)
def pad(grand_ave):
    array = np.array(grand_ave, dtype=object)
    max_sensors = max(matrix.shape[0] for matrix in array)
    padded = [np.pad(matrix, ((0, max_sensors - matrix.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)
              for matrix in grand_ave]
    return np.array(padded)

def interp(grand_ave):
    # interpolate function to have a list of interpolated coherence matrices with the same size
    array = np.array(grand_ave, dtype=object)
    # find the maximum number of sensors across all subjects
    max_sensors = max(matrix.shape[0] for matrix in array)
    
    # initialize a list to store interpolated coherence matrices
    interpolated_coherence = []
    
    # interpolate coherence matrices to have the same number of sensors
    for c in array:
        n_sensors = c.shape[0]
        
        # interpolate along the sensor axis
        x = np.arange(n_sensors)
        f = interpolate.interp1d(x, c, axis=0, kind='linear', fill_value='extrapolate')
        
        # create new axis for interpolated sensors
        new_x = np.linspace(0, n_sensors - 1, max_sensors)
        
        # interpolate coherence matrix
        interpolated = f(new_x)
        interpolated_coherence.append(interpolated)
        
    # average over all subjects axis
    inter_grand_ave = np.mean(interpolated_coherence, axis=0) 
    return inter_grand_ave
# %%
# average on subject axis and average while ignoring the nan we just created to have equal shape for al
# pad_grand_ave_coh = np.nanmean(pad(all_coherences), axis=0)
# pad_grand_ave_coh_left = np.nanmean(pad(all_coherences_left), axis=0)
# pad_grand_ave_coh_right = np.nanmean(pad(all_coherences_right), axis=0)

# # internally topomap function uses ascale of 1e13**2 in the data because units is grad
# # cancel out this scaling to keep our actual coherence values 
scaling = 1e13 ** 2

# padding
# grand_ave_coh = (pad_grand_ave_coh / scaling)
# grand_ave_coh_left = (pad_grand_ave_coh_left / scaling)
# grand_ave_coh_right = (pad_grand_ave_coh_right / scaling)

# interp
grand_ave_coh = (interp(all_coherences) / scaling)
grand_ave_coh_left = (interp(all_coherences_left) / scaling)
grand_ave_coh_right = (interp(all_coherences_right)/ scaling)
# %%
# we create a SpectrumArray object for visualising the topomap using the meg_info_max (204 sensors)
grand_ave_coh = mne.time_frequency.SpectrumArray(grand_ave_coh, meg_info_max, freqs=coh_freqs)
grand_ave_coh_left = mne.time_frequency.SpectrumArray(grand_ave_coh_left, meg_info_max, freqs=coh_freqs)
grand_ave_coh_right = mne.time_frequency.SpectrumArray(grand_ave_coh_right, meg_info_max, freqs=coh_freqs)

# %%
freqs_beta = {'beta': (13, 30)} 
dB = False

fig, ax = plt.subplots(1, 3, figsize=(8, 6))
grand_ave_coh.plot_topomap(bands=freqs_beta, res=300, cmap=(parula_map, True), dB=dB, axes=ax[0])
grand_ave_coh_left.plot_topomap(bands=freqs_beta, res=300, cmap=(parula_map, True), dB=dB, axes=ax[1])
grand_ave_coh_right.plot_topomap(bands=freqs_beta, res=300, cmap=(parula_map, True), dB=dB, axes=ax[2])

# remove colorbar defaut label
for cbar in fig.axes:
    if isinstance(cbar, plt.Axes):
        cbar.set_ylabel('') 
        
# set title for each axes
ax[0].set_title('Average ALL')
ax[1].set_title('Average left')
ax[2].set_title('Average right')

# save the grand average inside the coherence folder.
save_multipage('./coherence/sub-GrandAverage.pdf')
plt.show()

# %%