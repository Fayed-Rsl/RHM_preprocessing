# %%
# import the main libraries required for the preprocessing
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree
import mne 
import pandas as pd 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne_connectivity
from utils import parula_map, preprocess_raw_to_epochs, individual_lfpmeg_coh, plot_individual_coh_topo, plot_coh_topo, save_multipage

# %matplotlib qt 
    
# %%
# loop over all the fif file in session PeriOp, remove empty room and split 02 as it is read when we read split 01
bids_root = '/data/raw/hirsch/RestHoldMove_anon/'
subject_files = BIDSPath(root=bids_root, session='PeriOp', extension='.fif').match()
subject_files = [file for file in subject_files if 'noise' not in file.basename]
subject_files = [file for file in subject_files if 'split-02' not in file.basename]

# %%
fmin = 13
fmax = 30
freqs = np.arange(fmin, fmax+1, 1)
freqs_beta = {'beta': (fmin, fmax)} #freq up to beta (13, 30)
dB = True

for bids in subject_files:

    # read raw bids files and transform it into epochs 
    epochs = preprocess_raw_to_epochs(bids, downsample=200)
    
    # get the sampling freq
    sfreq = epochs.info['sfreq']
    
    # get the position of the sensor to split between ch meg left
    meg = epochs.copy().pick_types(meg='grad')
    lfp = epochs.copy().pick_types(eeg=True)
    
    # get the LFP channel names for right and left side
    ch_lfp_right = [ch for ch in lfp.ch_names if 'right' in ch]
    ch_lfp_left = [ch for ch in lfp.ch_names if 'left' in ch]
    
    # find the indices of the desired channel names for averaging later
    lfp_right_ind = [lfp.ch_names.index(name) for name in ch_lfp_right]
    lfp_left_ind = [lfp.ch_names.index(name) for name in ch_lfp_left]
    same_size = len(lfp_right_ind) == len(lfp_left_ind)
    print(f'LFP right and left are equal: {same_size}')
    
    # get the epochs data to compute connectivity in
    data = epochs.get_data(copy=True)
    
    # get the channel length for meg and lfp 
    n_meg_sensors =  len(meg.ch_names)
    n_lfp_sensors = len(lfp.ch_names) 
    
    # create indices: seeds and targets for connectivity measure so : get the index where it is LFP and MEG
    seeds  = mne.pick_types(epochs.info, eeg=True)# going from 0 to 202 MEG sensors as it in the epochs._data 
    targets = mne.pick_types(epochs.info, meg=True) # we want the connectivity between all sensors
    # n_connection indices[0] should be equal to n_lfp_sensors * n_meg_sensors
    indices = mne_connectivity.seed_target_indices(seeds, targets) 

    individual_coh_list = individual_lfpmeg_coh(seeds, targets, data, sfreq, fmin, fmax)
    

    coh_lfpmeg = mne_connectivity.spectral_connectivity_epochs(data, fmin=fmin, fmax=fmax, method='coh',
                mode='multitaper', sfreq=sfreq, indices=indices, n_jobs=-1)
    
    # plot individual topomap of all lfp contact
    plot_individual_coh_topo(individual_coh_list, lfp.ch_names, meg.info, freqs_beta, dB=dB, show=False)
    
    # make a fig that will have --> make the average for: all, left, right
    fig, ax = plt.subplots(1, 3, figsize=(8, 6)) 
    plot_coh_topo(coh=coh_lfpmeg, n_lfp=n_lfp_sensors, n_meg=n_meg_sensors, meg_info=meg.info, freqs_beta=freqs_beta,
                  average_type='all', ax=ax[0], dB=dB)
    
    # plot average left side
    plot_coh_topo(coh=coh_lfpmeg, n_lfp=n_lfp_sensors, n_meg=n_meg_sensors, meg_info=meg.info, freqs_beta=freqs_beta,
                  average_type='side', lfp_ind=lfp_left_ind, ax=ax[1], dB=dB)
    
    # plot average_right side 
    plot_coh_topo(coh=coh_lfpmeg, n_lfp=n_lfp_sensors, n_meg=n_meg_sensors, meg_info=meg.info, freqs_beta=freqs_beta,
                  average_type='side', lfp_ind=lfp_right_ind, ax=ax[2], dB=dB)
    
    ax[0].set_title('Average ALL')
    ax[1].set_title('Average left')
    ax[2].set_title('Average right')
    
    # save the power for further group analysis | remove the extension
    coh_basename = bids.copy().update(suffix='coh', extension='.', split=None, check=False).basename.replace('.', '')
    img_basename = bids.copy().update(suffix='img', extension='.pdf', split=None, check=False).basename
    
    sub_path = Path(f'coherence/{bids.subject}/')
    sub_path.mkdir(parents=True, exist_ok=True)     
    
    coh_path = Path(f'{sub_path}/{coh_basename}')
    img_path = Path(f'{sub_path}/{img_basename}')
    

    # save the coherence between lfp and meg for grand average group
    coh_lfpmeg.save(coh_path)
    save_multipage(img_path)
    plt.show() # show the plot the
    plt.close('all')

# minor checks 

# %%
# get the data in a list of the ind coh ... 
# data_ave = np.array(data_ave)

# # %%
# coh_data = coh_lfpmeg.get_data() 
# coh_freqs = np.array(coh_lfpmeg.freqs)
# # reshape the data according to (N_lfp * N_meg * N_freqs)
# coh_data = coh_data.reshape(n_lfp_sensors, n_meg_sensors, len(coh_freqs))

# # average the connectivity on LFP axis channel, average type: either all LFP, only left or right side
# # average_type = 'all'
# coh_average = np.mean(coh_data, axis=0)
# data_average = np.mean(data_ave, axis=0)


# coh_spec = mne.time_frequency.SpectrumArray(coh_average, meg.info, freqs=coh_freqs)
# data_spec = mne.time_frequency.SpectrumArray(data_average, meg.info, freqs=coh_freqs)

# # %%
# fig, ax = plt.subplots(1, 2, figsize=(12, 8))
# coh_spec.plot_topomap(bands=freqs_beta, res=300, cmap=(parula_map, True), dB=dB, axes=ax[0], units='conn')
# ax[0].set_title('Doing connectivity one shot then reshape')

# data_spec.plot_topomap(bands=freqs_beta, res=300, cmap=(parula_map, True), dB=dB, axes=ax[1], units='conn')
# ax[1].set_title('Doing Connectivity one by one and concatenate')



# %%
