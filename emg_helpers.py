# %%
# import the main libraries required for the preprocessing
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree
import mne 
import pandas as pd 
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statannotations.Annotator import Annotator 
%matplotlib qt 

# %%
bids_root = '/data/raw/hirsch/RestHoldMove_anon/'


# %matplotlib qt 
# %%
print_dir_tree(bids_root)
# %% 
# setting of a specific file
session = 'PeriOp'
subject = 'QZTsn6'
datatype = 'meg'
task = 'MoveL'
acq = 'MedOff'
run = '1'

bids_path = BIDSPath(root=bids_root, session=session, subject=subject, task=task, acquisition=acq,
                     datatype=datatype, run=run, extension='.fif')

print(bids_path.match())
# %%
raw = read_raw_bids(bids_path)

# if there is any bad channel remove them
if len(raw.info['bads']) > 0: 
    # check bad channels (MEG and LFP)
    print(raw.info['bads'])

    # get all channel types
    channel_to_pick = {'emg': True}

    # pick all channels types and exclude the bads one
    raw.pick_types(**channel_to_pick)

# load the data
raw.load_data()

# apply a 1Hz high pass firwin filter to remove slow drifts
raw.filter(l_freq=1, h_freq=None, method='fir', n_jobs=-1, picks=['emg'])
#%%
emgs = raw.get_data()
sfreq = raw.info['sfreq']

# %%
ch_names = raw.ch_names

# apply a 1Hz high pass firwin filter to remove slow drifts
# raw.filter(l_freq=1, h_freq=None, method='fir', n_jobs=-1)


emg_right = emgs[:2]
emg_left = emgs[2:]

mean_right = np.mean(emg_right)
mean_left = np.mean(emg_left)

rms_right = np.sqrt(np.mean(emg_right**2))
rms_left = np.sqrt(np.mean(emg_left**2))

# selected_emg =  emg_left if mean_left > mean_right else  emg_right  
# selected_emg_label = 'Left' if mean_left > mean_right else 'Right' 
# print(selected_emg_label)

selected_emg =  emg_left if rms_left > rms_right else  emg_right  
selected_emg_label = 'Left' if rms_left > rms_right else 'Right' 

if selected_emg_label == 'Right':

    ch_names = ch_names[:2]
else:
    ch_names = ch_names[2:]
    
print(selected_emg_label, ch_names)
# need to selected appropriate timing of the raw before computing the EMG
# %%
# take the mean of both channels 
x_right = np.mean(emg_right, axis=0)
x_left = np.mean(emg_left, axis=0)

# %%
# selected_emg = raw.copy().pick_channels(ch_names)

# spectrum2 = selected_emg.compute_psd(method='welch', fmin=0, n_jobs=-1)
#%%
emg_list = [x_right, x_left]
emg_label = ['right', 'left']
for emg, label in zip(emg_list, emg_label):
    # remove freq 1 because we filtered the 1 hz
    psd, freqs = mne.time_frequency.psd_array_welch(emg, sfreq=sfreq, fmin=1, fmax=200, n_fft=2048)
    # plt.figure()
    plt.plot(freqs, psd, label=label)

plt.title(f'Selected emg was {selected_emg_label} and task {task}')
plt.legend()
plt.show()
# %%
# get the onset and the duration for task ('MoveL') event
# resting state period marked and the selected task.
# Note that it is also possible to extend the rest period to everything that it is not marked as the task.
# But here we only include the start rest period not the rest period between the task aswell.
tasks = ['rest'] + [task]
for t in tasks:
    task_segments_onset = raw.annotations.onset[raw.annotations.description == t]
    task_segments_duration = raw.annotations.duration[raw.annotations.description == t]
    
    # substract the first_samp delay in the task onset
    task_segments_onset = task_segments_onset - (raw.first_samp / raw.info['sfreq'])
    
    # loop trough the onset and duration to get only part of the raw that are in the task
    for i, (onset, duration) in enumerate(zip(task_segments_onset, task_segments_duration)):
        # if it is the first onset, initialise the raw object storing all of the task segments
        if t == task:
            if i == 0:
                task_segments = raw.copy().crop(tmin=onset, tmax=onset+duration)
            # otherwise, append the segments to the existing raw object
            else:
                task_segments.append([raw.copy().crop(tmin=onset, tmax=onset+duration)])
        else:
            if i == 0:
                rest_segments = raw.copy().crop(tmin=onset, tmax=onset+duration)
            # otherwise, append the segments to the existing raw object
            else:
                rest_segments.append([raw.copy().crop(tmin=onset, tmax=onset+duration)])


# %%

def get_selected_emg(emgs, ch_names, side):
    
    # find the side automatically using RMS 
    emg_left = emgs[2:]
    emg_right = emgs[:2]
    
    if side == 'find':
        rms_right = np.sqrt(np.mean(emg_right**2))
        rms_left = np.sqrt(np.mean(emg_left**2))
        
        selected_emg =  emg_left if rms_left > rms_right else  emg_right
        selected_emg_mean = np.mean(selected_emg, axis=0) # average of the 2 EMG channels
        selected_emg_label = 'left' if rms_left > rms_right else 'right' 
        
    elif side == 'left':
        selected_emg_mean =  np.mean(emg_left, axis=0)
        selected_emg_label = side 
    elif side == 'right':
        selected_emg_mean =  np.mean(emg_right, axis=0)
        selected_emg_label = side 

    if selected_emg_label == 'right':
        ch_names = ch_names[:2]
    else:
        ch_names = ch_names[2:]
        
    print(selected_emg_label, ch_names)
    return selected_emg_mean, selected_emg_label

    
def get_emg_power(raw, ch_names, side='find'):
    emgs = raw.get_data()
    sfreq = raw.info['sfreq'] 
    selected_emg, selected_emg_label = get_selected_emg(emgs, ch_names, side)

    # # take the mean of both channels
    # x_right = np.mean(emg_right, axis=0)
    # x_left = np.mean(emg_left, axis=0)
    
    # emg_list = [x_right, x_left]
    # emg_label = ['right', 'left']
    # plt.figure()
    # for emg, label in zip(emg_list, emg_label):
    # remove freq 1 because we filtered the 1 hz
    psd, freqs = mne.time_frequency.psd_array_welch(selected_emg, sfreq=sfreq, fmin=2, fmax=200, n_fft=2048)
    
    return psd, freqs, selected_emg_label

def plot_power(psd, freqs, side, condition, color):
    # plt.figure()
    plt.plot(freqs, psd, label=f'{condition}', alpha=0.5, color=color)
    plt.title(f'Selected EMG during Rest Hold Move condition')
    plt.legend()
    plt.show()

# %%
psd_task, freqs, side = get_emg_power(task_segments, raw.ch_names, side='find')
psd_rest, _, _ = get_emg_power(rest_segments, raw.ch_names, side=side)

plot_power(psd_task, freqs, side, condition=task, color='blue')
plot_power(psd_rest, freqs, side, condition='rest', color='orange')

# np.savez(f'freqs.npz', )
emg_path = f'emgs/{subject}_emg_data.npz'
np.savez(emg_path, freqs=freqs, psd_rest=psd_rest, psd_task=psd_task)

# %%
emg_load = np.load(emg_path)

# save the power for further group analysis


# get_power_and_plot(rest_segments, rest, 'rest', 'orange')
# get_power_and_plot(task_segments, task, 'task', 'blue')

# plt.show()

# mean_psd = psd.mean(0)
# mean_psd = mean_psd.reshape(1, 205)


# %%
# info = mne.create_info(ch_names, sfreq=sfreq, ch_types='eeg') # pass it as EEG to create the info even though it is EMG here 
info = mne.create_info(['EMG'], sfreq=sfreq, ch_types='eeg') # pass it as EEG to create the info even though it is EMG here 

# spectrum = mne.time_frequency.SpectrumArray(data=psd, freqs=freqs, info=info)
spectrum = mne.time_frequency.Spectrum(selected_emg_psd, method='welch', fmin=0, fmax=200)


# %%
spectrum.plot()


# %%
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