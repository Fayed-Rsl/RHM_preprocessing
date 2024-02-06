# This script has been made in order to quickly start analysing the MEG-LFP Dataset refering to this article: doi...
# For example purposes, preprocessing has been implemented only for a specific: subject, session, task, acquisition.
# One would need to adapt the following code in order to make analyse on the whole dataset as it is examplified in run_coherence.py
# Feel free to modify the script as pleased.
# INFORMATION: You would need to change the bids_root to make it match with the location of the bids_dataset
#----------------------------------------------------------------------------#

# import the main libraries required for the preprocessing
from mne_bids import BIDSPath, read_raw_bids
import mne 
import matplotlib.pyplot as plt
import pandas as pd 
import mne_connectivity

# import homemade function 
from technical_validation_utils import (get_raw_condition, plot_individual_coh_topo, plot_coh_topo)
# %%
# define the bids root
bids_root = '/data/raw/hirsch/RestHoldMove_anon/'

# setting for loading a specific file
session = 'PeriOp'
subject = 'QZTsn6'
datatype = 'meg'
task = 'MoveL'
acquisition = 'MedOff'
run = '1'

bids_path = BIDSPath(root=bids_root, session=session, subject=subject, datatype=datatype, task=task,
                     acquisition=acquisition, run=run, extension='.fif')

print(bids_path.match())
# %%
# read the raw data
raw = read_raw_bids(bids_path)

# print if there is any bad channel
if len(raw.info['bads']) > 0: 
    print(raw.info['bads'])

# get all channel types, here we ignore eog and stim channel
channel_to_pick = {'meg':'grad', 'eeg': True, 'emg': True, 'eog': False, 'stim': False}

# pick all of the channels types and exclude the one marked as bads
raw.pick_types(**channel_to_pick, exclude='bads')

# create a bids path for the montage, check=False because this is not a standard bids file
montage_path = BIDSPath(root=bids_root, session=session, subject=subject,
                        datatype='montage', extension='.tsv', check=False)

# get the montage tsv bids file 
montage = montage_path.match()[0]

# read the file using pandas
df = pd.read_csv(montage, sep='\t')

# create a dictionary mapping old names to new names for right and left channels
montage_mapping = {row['right_contacts_old']: row['right_contacts_new'] for idx, row in df.iterrows()} # right
montage_mapping.update({row['left_contacts_old']: row['left_contacts_new'] for idx, row in df.iterrows()}) # left

# remove in the montage mapping the channels that are not in the raw anymore (because they were marked as bads)
montage_mapping  = {key: value for key, value in montage_mapping.items() if key in raw.ch_names }
print(montage_mapping)

# rename the channels using the montage mapping scheme
raw.rename_channels(montage_mapping)

# get list of lfp names and their side for bipolar contact
lfp_name = [name for name in raw.ch_names if 'LFP' in name]
lfp_right = [name for name in lfp_name if 'right' in name]
lfp_left = [name for name in lfp_name if 'left' in name]

# get LFP-side-contact then add the next contact number to the name to create the new bipolar name
# e.g. --> LFP-right-0 and LFP-right-1 will become LFP-right-01
if len(lfp_right) > 1: # we need at least 2 contact to create a bipolar scheme
    bipolar_right = [f'{lfp_right[i]}{lfp_right[i+1][-1]}' for i in range(len(lfp_right)-1)]
    ref_right = True
else:
    ref_right = False # else do no re-reference this side

if len(lfp_left) > 1:
    bipolar_left = [f'{lfp_left[i]}{lfp_left[i+1][-1]}' for i in range(len(lfp_left)-1)]
    ref_left = True
else: 
    ref_left = False

# from now on we need to load the data to apply filter and re-referencing
raw.load_data()

# set bipolar reference scheme for each respective side 
if ref_right:
    raw = mne.set_bipolar_reference(raw, anode=lfp_right[:-1], cathode=lfp_right[1:], ch_name=bipolar_right)
    
if ref_left:
    raw = mne.set_bipolar_reference(raw, anode=lfp_left[:-1], cathode=lfp_left[1:], ch_name=bipolar_left)

# apply a 1Hz high pass firwin filter to remove slow drifts
raw.filter(l_freq=1, h_freq=None, method='fir', n_jobs=-1)

# after filtering, now do downsample to 200 from 2000. NB: downsample after filter so no aliasing introduced
raw.resample(200)

# prepare the conditions to crop the data and get the rest_segments and task_segments
conditions = ['rest'] + [bids_path.task] # rest + task (hold or move)

# if the task is only 'Rest' then keep the raw segments as it is and ignore the task_segments
# please refer to the function get_raw_condition defined in the technical_validation_utils.py file 
if bids_path.task == 'Rest': # only true for 2 subject
    rest_segments, _ = get_raw_condition(raw, ['rest'])
# otherwise, crop the data and get the rest_segments and the task_segments
else:
    rest_segments, task_segments = get_raw_condition(raw, conditions)

# %%
# from now on we will focus only in the resting state segment we just cropped
# create epochs of fixed length 2 seconds and 50% overlap.
epochs = mne.make_fixed_length_epochs(rest_segments, duration=2, overlap=1, preload=True)

# get the sampling freq
sfreq = epochs.info['sfreq']

# create epochs with only meg or lfp
meg = epochs.copy().pick_types(meg='grad')
lfp = epochs.copy().pick_types(eeg=True)

# get the LFP channel names for right and left side with the bipolar scheme
lfp_right = [name for name in lfp.ch_names if 'right' in name]
lfp_left = [name for name in lfp.ch_names if 'left' in name]
    
# find the indices of the desired channel names for averaging later
lfp_right_ind = [lfp.ch_names.index(name) for name in lfp_right]
lfp_left_ind = [lfp.ch_names.index(name) for name in lfp_left]

# get the epochs data to compute connectivity
data = epochs.get_data(copy=True)

# get the channel length for meg and lfp 
n_meg_sensors =  len(meg.ch_names)
n_lfp_sensors = len(lfp.ch_names)

# create indices: seeds and targets for connectivity measure between LFP and all MEG sensors
seeds  = mne.pick_types(epochs.info, eeg=True) # indices of the LFP
targets = mne.pick_types(epochs.info, meg=True) # indices of the MEG
# n_connection indices[0] should be equal to n_lfp_sensors * n_meg_sensors
indices = mne_connectivity.seed_target_indices(seeds, targets) 

# here we compute connectivity in the beta band only
fmin = 13
fmax = 30
freqs_beta = {'beta': (fmin, fmax)} # freq up to beta (13, 30) 

# compute connectivity between ALL contacts and all MEG sensors should be (n_lfp*n_meg)
coh_lfpmeg = mne_connectivity.spectral_connectivity_epochs(data, fmin=fmin, fmax=fmax, method='coh',
            mode='multitaper', sfreq=sfreq, indices=indices, n_jobs=-1)

# %%
# plot individual topomap of all individual lfp contact, please refer to the plot function in the utils
plot_individual_coh_topo(coh=coh_lfpmeg, lfp_ch_names=lfp.ch_names, n_lfp=n_lfp_sensors, n_meg=n_meg_sensors,
                         meg_info=meg.info, freqs_beta=freqs_beta)

# %%
# now let's make a figure that will have --> the average for: all (1), left (2), right (3)
fig, ax = plt.subplots(1, 3, figsize=(8, 6))

# plot average for all contact
plot_coh_topo(coh=coh_lfpmeg, n_lfp=n_lfp_sensors, n_meg=n_meg_sensors, meg_info=meg.info, freqs_beta=freqs_beta,
              average_type='all', ax=ax[0])

# plot average for left side only if there is at least one LFP
if len(lfp_left_ind) > 1:
    plot_coh_topo(coh=coh_lfpmeg, n_lfp=n_lfp_sensors, n_meg=n_meg_sensors, meg_info=meg.info, freqs_beta=freqs_beta,
                  average_type='side', lfp_ind=lfp_left_ind, ax=ax[1])
else:
    fig.delaxes(ax[1]) # remove the ax from the plot

# plot average for right side only if there is at least one LFP
if len(lfp_right_ind) > 1:
    plot_coh_topo(coh=coh_lfpmeg, n_lfp=n_lfp_sensors, n_meg=n_meg_sensors, meg_info=meg.info, freqs_beta=freqs_beta,
                  average_type='side', lfp_ind=lfp_right_ind, ax=ax[2])
else:
    fig.delaxes(ax[2]) # remove the ax from the plot
        
# set title for each axes
ax[0].set_title('Average All')
ax[1].set_title('Average Left')
ax[2].set_title('Average Right')
    
# remove colorbar defaut label created for grad (ft/cm² ...) since we are working with connectivity values here
for cbar in fig.axes:
    if isinstance(cbar, plt.Axes):
        cbar.set_ylabel('') 


# END OF PREPROCESSING TEMPLATE
