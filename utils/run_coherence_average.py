# import the main libraries required for the preprocessing
import mne 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne_connectivity
import os 
import pickle

# import homemade function 
from technical_validation_utils import (parula_map, scaling, interp, subjects)

mne.set_log_level(verbose='CRITICAL') # reduce verbose output
# %matplotlib qt

# %%
# NB: run_coherence.py must has been runned before running the average
coherence_dir = [Path('./coherence/') / sub for sub in subjects]
assert all(file.exists() for file in coherence_dir), 'Please run_coherence.py. Path has not been created yet.'
freqs_beta = {'beta': (13, 30)} 
dB = False

# store variable for coherences 
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
        if len(meg_info.ch_names) == 204 and got_info != True: # hardcoded 204 sensors as we know it is the max
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
        coherences_left.append(coh_average_left)
        coherences_right.append(coh_average_right)

    # average on files axis and append it to the grand average list
    all_coherences_left.append(np.mean(coherences_left, axis=0))
    all_coherences_right.append(np.mean(coherences_right, axis=0))

# %%
# internally topomap function uses ascale of 1e13**2 in the data because units is grad
# cancel out this scaling to keep our actual coherence values 

# interpolate missing sensors and rescale it for the coherences list 
grand_ave_coh_left = (interp(all_coherences_left) / scaling)
grand_ave_coh_right = (interp(all_coherences_right)/ scaling)
# %%
# we create a SpectrumArray object for visualising the topomap using the meg_info_max (204 sensors)
grand_ave_coh_left = mne.time_frequency.SpectrumArray(grand_ave_coh_left, meg_info_max, freqs=coh_freqs)
grand_ave_coh_right = mne.time_frequency.SpectrumArray(grand_ave_coh_right, meg_info_max, freqs=coh_freqs)

# %%
# plot grand average for each sides 
# ------------------------------------
fig, ax = plt.subplots(figsize=(8, 6)) # LEFT
grand_ave_coh_left.plot_topomap(bands=freqs_beta, res=300, cmap=(parula_map, True), dB=dB, axes=ax, cbar_fmt='%0.2f')
[cbar.set_ylabel('') for cbar in fig.axes if isinstance(cbar, plt.Axes)]
ax.set_title('')
fig.savefig('./figures/sub-GrandAverageCOH-left.jpg', dpi=300)
plt.show()

# ------------------------------------
fig, ax = plt.subplots(figsize=(8, 6)) # RIGHT
grand_ave_coh_right.plot_topomap(bands=freqs_beta, res=300, cmap=(parula_map, True), dB=dB, axes=ax, cbar_fmt='%0.2f')

# remove colorbar defaut label and title and save 
[cbar.set_ylabel('') for cbar in fig.axes if isinstance(cbar, plt.Axes)]
ax.set_title('')
fig.savefig('./figures/sub-GrandAverageCOH-right.jpg', dpi=300)
plt.show()
# %%