# %%
# import the main libraries required for the preprocessing
from mne_bids import BIDSPath
import mne 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne_connectivity
import pickle

# import homemade function 
from technical_validation_utils import (preprocess_raw_to_epochs, individual_lfpmeg_coh,
                                        plot_individual_coh_topo, plot_coh_topo, save_multipage)

mne.set_log_level(verbose='CRITICAL') # reduce verbose output
# %matplotlib qt # interactive plot
    
# %%
# create a list of subjects to loop over all the fif file in session PeriOp
# remove empty room and split 02 as it is read when we read split 01
bids_root = '/data/raw/hirsch/RestHoldMove_anon/'
subject_files = BIDSPath(root=bids_root, session='PeriOp', extension='.fif').match()
subject_files = [file for file in subject_files if 'noise' not in file.basename]
subject_files = [file for file in subject_files if 'split-02' not in file.basename]

# %%
# set the frequency for the coherence to compute 
fmin = 13
fmax = 30
freqs = np.arange(fmin, fmax+1, 1) # [13, ..., 30]
freqs_beta = {'beta': (fmin, fmax)} # freq up to beta (13, 30) 
dB = True # log values for the plot

# loop trough all the subject_files
for n_file, bids in enumerate(subject_files):
    # create filename to save the power and indices for further group analysis and image for visual check
    coh_basename = bids.copy().update(suffix='coh', extension='.', split=None, check=False).basename.replace('.', '') # remove extension
    ind_basename = bids.copy().update(suffix='ind', extension='.pkl', split=None, check=False).basename
    img_basename = bids.copy().update(suffix='img', extension='.pdf', split=None, check=False).basename
    
    # subject path, if the directory does not already exist make one
    sub_path = Path(f'./coherence/sub-{bids.subject}/')
    sub_path.mkdir(parents=True, exist_ok=True)     
    
    # combine full fname for saving 
    coh_path = Path(f'{sub_path}/{coh_basename}')
    ind_path = Path(f'{sub_path}/{ind_basename}')
    img_path = Path(f'{sub_path}/{img_basename}')
    
    # check if the files already exist, then the analyse has already been made then skip to the next file
    if coh_path.exists() and img_path.exists():
        # print(f'{bids} already processed')
        continue
    
    # read raw bids files and transform it into epochs 
    epochs = preprocess_raw_to_epochs(bids, downsample=200)
    
    # if the epochs is None because there is no resting state in the file then continue and go to the next 
    if epochs is None:
        continue
    
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
    # print(f'LFP right and left are equal: {same_size}')
    
    # get the epochs data to compute connectivity
    data = epochs.get_data(copy=True)
    
    # get the channel length for meg and lfp 
    n_meg_sensors =  len(meg.ch_names)
    n_lfp_sensors = len(lfp.ch_names) 
    
    # create indices: seeds and targets for connectivity measure so : get the index where it is LFP and MEG
    seeds  = mne.pick_types(epochs.info, eeg=True)# going from 0 to 202 MEG sensors as it in the epochs._data 
    targets = mne.pick_types(epochs.info, meg=True) # we want the connectivity between all sensors
    # n_connection indices[0] should be equal to n_lfp_sensors * n_meg_sensors
    indices = mne_connectivity.seed_target_indices(seeds, targets) 

    # compute connectivity between individual contact and all MEG sensors. (1*202)
    individual_coh_list = individual_lfpmeg_coh(seeds, targets, data, sfreq, fmin, fmax)
    
    # compute connectivity between ALL contacts and all MEG sensors (8*202)
    coh_lfpmeg = mne_connectivity.spectral_connectivity_epochs(data, fmin=fmin, fmax=fmax, method='coh',
                mode='multitaper', sfreq=sfreq, indices=indices, n_jobs=-1)
    
    # plot individual topomap of all individual lfp contact
    plot_individual_coh_topo(individual_coh_list, lfp.ch_names, meg.info, freqs_beta, dB=dB, show=False)
    
    # make a fig that will have --> the average for: all (1), left (2), right (3)
    fig, ax = plt.subplots(1, 3, figsize=(8, 6))
    
    # average for all contact
    plot_coh_topo(coh=coh_lfpmeg, n_lfp=n_lfp_sensors, n_meg=n_meg_sensors, meg_info=meg.info, freqs_beta=freqs_beta,
                  average_type='all', ax=ax[0], dB=dB)
    
    # average for left side
    if len(lfp_left_ind) > 1:
        plot_coh_topo(coh=coh_lfpmeg, n_lfp=n_lfp_sensors, n_meg=n_meg_sensors, meg_info=meg.info, freqs_beta=freqs_beta,
                      average_type='side', lfp_ind=lfp_left_ind, ax=ax[1], dB=dB)
    else:
        fig.delaxes(ax[1]) # remove the ax from the plot

    # average for right side 
    if len(lfp_right_ind) > 1:
        plot_coh_topo(coh=coh_lfpmeg, n_lfp=n_lfp_sensors, n_meg=n_meg_sensors, meg_info=meg.info, freqs_beta=freqs_beta,
                      average_type='side', lfp_ind=lfp_right_ind, ax=ax[2], dB=dB)
    else:
        fig.delaxes(ax[2]) # remove the ax from the plot
            
    # set title for each axes
    ax[0].set_title('Average ALL')
    ax[1].set_title('Average left')
    ax[2].set_title('Average right')
    
    # save the coherence between lfp and meg for grand average group
    coh_lfpmeg.save(coh_path)
    
    # save the seeds, targets, left, right, n_meg, n_lfp
    save_coh_info = {'n_lfp_sensors':n_lfp_sensors,
                    'n_meg_sensors':n_meg_sensors,
                    'meg_info':meg.info,
                    'seeds':seeds,
                    'targets':targets,
                    'lfp_names':lfp.ch_names,
                    'lfp_left_ind':lfp_left_ind,
                    'lfp_right_ind':lfp_right_ind}
      
    
    with open(ind_path, 'wb') as f: 
        pickle.dump(save_coh_info, f)

    # save all the plots produce for this bids file inside a PDF for further check.
    save_multipage(img_path)
    
    # show the plot inside the IDE and close.
    plt.show() 
    plt.close('all')
    
    print(f'Done with file n°{n_file} | {bids.basename}')

print('Coherence MEG-STN LFP Done.')
# %%