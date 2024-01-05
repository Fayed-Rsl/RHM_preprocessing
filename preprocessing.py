# This script has been made in order to quickly start analysing the MEG-LFP Dataset refering to this article: doi...
# For example purposes, preprocessing has been implemented only for a specific: subject, session, task, acquisition.
# One would need to adapt the following code in order to make the preprocessing on the whole dataset.
# Feel free to modify it.

# %%
# import the main libraries required for the preprocessing
import os
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree
import mne 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from pathlib import Path
from scipy.io import loadmat


# %%
bids_root = '/data/raw/hirsch/RestHoldMove_anon2/'
print_dir_tree(bids_root)
# %% 
# setting of a specific file
session = 'PeriOp'
subject = 'hnetKS'
datatype = 'meg'
task = 'MoveL'
acq = 'MedOff'
run = '1'

bids_path = BIDSPath(root=bids_root, session=session, subject=subject, task=task, acquisition=acq,
                     datatype=datatype, run=run, extension='.fif')

print(bids_path.match())

# %%
# read the raw data
raw = read_raw_bids(bids_path)


# %%
# load the data
raw.load_data()

# apply a 1Hz high pass firwin filter to remove slow drifts
raw.filter(l_freq=1, h_freq=None, method='fir', n_jobs=-1)

# if there is any bad channel remove them
if len(raw.info['bads']) > 0: 
    # check bad channels (MEG and LFP)
    print(raw.info['bads'])

    # get all channel types
    channel_to_pick = {'meg':True, 'eeg': True, 'eog': True, 'stim': True, 'emg': True}

    # pick all channels types and exclude the bads one
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

# rename the channels using the mapping
raw.rename_channels(montage_mapping)


# plot the EEG before and after setting the reference to observe the difference 
# raw_ref = raw.copy().set_eeg_reference(ref_channels='average')
# scalings = {'eeg':'2e-4'}
# raw.copy().pick_types(eeg=True).plot(scalings=scalings)
# raw_ref.copy().pick_types(eeg=True).plot(scalings=scalings)


# set eeg reference using average scheme
raw.set_eeg_reference(ref_channels='average')


# get the onset and the duration for task ('MoveL') event
task_segments_onset = raw.annotations.onset[raw.annotations.description == task]
task_segments_duration = raw.annotations.duration[raw.annotations.description == task]

# substract the first_samp delay in the task onset
task_segments_onset = task_segments_onset - (raw.first_samp / raw.info['sfreq'])

# loop trough the onset and duration to get only part of the raw that are in the task
for i, (onset, duration) in enumerate(zip(task_segments_onset, task_segments_duration)):
    # if it is the first onset, initialise the raw object storing all of the task segments
    if i == 0:
        raw_segments = raw.copy().crop(tmin=onset, tmax=onset+duration)
    # otherwise, append the segments to the existing raw object
    else:
        raw_segments.append([raw.copy().crop(tmin=onset, tmax=onset+duration)])
    

# get the events from the raw annotations
events, events_id = mne.events_from_annotations(raw_segments)

# make fixed_events assuming that there is one move every 2 seconds with an overlap of 1 sec
events = mne.make_fixed_length_events(raw_segments, duration=2, overlap=1)
events_id = {task:1}

# epochs the data and reject the bad ones by annotation
epochs = mne.Epochs(raw_segments, events, events_id, preload=True, reject_by_annotation=True)

# standard tfr 
frequencies = np.arange(7, 30, 3)
power = mne.time_frequency.tfr_morlet(
    epochs, n_cycles=2, return_itc=False, freqs=frequencies, decim=3, n_jobs=-1)
power.plot(['LFP-right-0'])

# %% UPDRS EXAMPLE
# load UPDRS dataset 
updrs_off = Path(bids_root) / 'participants_updrs_off.tsv'
updrs_on = Path(bids_root) / 'participants_updrs_on.tsv'

off = pd.read_csv(updrs_on, sep='\t')
on = pd.read_csv(updrs_off, sep='\t')
# %% HEADMODEL EXAMPLE 
# create a bids path for the headmodel, check=False because this is not a standard bids file
headmodel_path = BIDSPath(root=bids_root, session=session, subject=subject,
                        datatype='headmodel', extension='.mat', check=False)
# get the path
headmodel_file = headmodel_path.match()[0]

# load the headmodel.mat file
headmodel = loadmat(headmodel_file)



