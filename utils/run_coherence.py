# import the main libraries required for the preprocessing
import mne 
from pathlib import Path
import matplotlib.pyplot as plt
import mne_connectivity
import pickle

# import homemade function 
from technical_validation_utils import (subject_files, preprocess_raw_to_epochs,
                                        plot_individual_coh_topo, plot_coh_topo, save_multipage)

mne.set_log_level(verbose='CRITICAL') # reduce verbose output
# %%
# set the frequency for the coherence to compute 
fmin = 13
fmax = 30
freqs_beta = {'beta': (fmin, fmax)} # freq up to beta (13, 30) 
dB = False # log values for the plot

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
    
    # create epochs with only meg or lfp
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
    
    # compute connectivity between ALL contacts and all MEG sensors (8*202)
    coh_lfpmeg = mne_connectivity.spectral_connectivity_epochs(data, fmin=fmin, fmax=fmax, method='coh',
                mode='multitaper', sfreq=sfreq, indices=indices, n_jobs=-1)
    
    # plot individual topomap of all individual lfp contact
    plot_individual_coh_topo(coh=coh_lfpmeg, lfp_ch_names=lfp.ch_names, n_lfp=n_lfp_sensors, n_meg=n_meg_sensors, meg_info=meg.info, freqs_beta=freqs_beta, dB=dB, show=False)
    
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
        
    # remove colorbar defaut label created for grad (ft/cm² ...)
    for cbar in fig.axes:
        if isinstance(cbar, plt.Axes):
            cbar.set_ylabel('') 
    
    # save the coherence between lfp and meg for grand average group
    coh_lfpmeg.save(coh_path)
    
    # save the coherence information seeds, targets, left, right, n_meg, n_lfp ... for the grand average
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