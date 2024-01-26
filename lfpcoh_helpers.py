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
import mne_connectivity
# %matplotlib qt 

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
task = 'HoldL'
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

# load the data
raw.load_data()

# set eeg reference using average scheme
raw.set_eeg_reference(ref_channels='average')

# apply a 1Hz high pass firwin filter to remove slow drifts
raw.filter(l_freq=1, h_freq=None, method='fir', n_jobs=-1)

# raw.resample(500)

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

freqs = np.arange(13, 31, 1)
downsample = 200
rest_segments.resample(downsample)
epochs_rest = mne.make_fixed_length_epochs(rest_segments, duration=2, overlap=1, preload=True)
# events_id_rest = {'rest':1}

task_segments.resample(downsample)
epochs_task = mne.make_fixed_length_epochs(task_segments, duration=2, overlap=1)
# events_id_task = {task:1}


# %%
import scipy
# coh = scipy.signal.coherence(meg._data.T, lfp._data.T, fs=downsample)
cofreq, coh = scipy.signal.coherence(meg._data[:8, : ], lfp._data, fs=downsample)


# %%
meglfp_epochs = epochs_rest.copy().pick_types(meg='grad', eeg=True) 
csd = mne.time_frequency.csd_multitaper(meglfp_epochs, fmin=13, fmax=31, n_jobs=-1)
# mne.viz.plot_topomap(csd._data, meglfp_epochs.copy().set_channel_types(dict(zip(lfp.ch_names, ['grad']*8))).info)
# %%
# get the MEG gradiometer and EEG STN-LFP from both dataset
meg = rest_segments.copy().pick_types(meg='grad')
lfp = rest_segments.copy().pick_types(eeg=True)


data_time = np.vstack((meg._data, lfp._data))
data_time = data_time[np.newaxis, :, :]

ch_lfp = epochs_rest.copy().pick_types(eeg=True).ch_names
ch_lfp_right = [ch for ch in ch_lfp if 'right' in ch]
ch_lfp_left = [ch for ch in ch_lfp if 'left' in ch]

meg_epochs = epochs_rest.copy().pick_types(meg='grad')
lfp_epochs = epochs_rest.copy().pick_types(eeg=True)
lfp_right_epochs = epochs_rest.copy().pick_channels(ch_lfp_right)
lfp_left_epochs = epochs_rest.copy().pick_channels(ch_lfp_left)
# %%
sfreq = rest_segments.info['sfreq']

# Get the data from epochs
meg_data_epochs = meg_epochs.get_data()
lfp_data_epochs = lfp_epochs.get_data()

# epochs = mne.concatenate_epochs([meg_epochs, lfp_epochs])

# Combine MEG and LFP data for connectivity analysis
data_epochs_right = np.concatenate((meg_epochs._data, lfp_right_epochs._data), axis=1)
data_epochs_left = np.concatenate((meg_epochs._data, lfp_left_epochs._data), axis=1)

info = rest_segments.copy().pick_types(meg='grad', eeg=True) 
info_l = info.copy().pick_channels(meg_epochs.ch_names + lfp_left_epochs.ch_names)
info_r = info.copy().pick_channels(meg_epochs.ch_names + lfp_right_epochs.ch_names)


# %%
# con_epoch2 = mne_connectivity.spectral_connectivity_epochs(data=data, method='coh', sfreq=sfreq, n_jobs=-1)
test = rest_segments.copy().pick_types(meg='grad')
sensor_pos = test.info['chs']
# Identify sensors on the left and right
left_sensors = [ch_name for i, ch_name in enumerate(test.ch_names) if sensor_pos[i]['loc'][0] < 0]
right_sensors = [ch_name for i, ch_name in enumerate(test.ch_names) if sensor_pos[i]['loc'][0] > 0]

# Create separate raw objects for left and right sensors
# raw_left = raw.copy().pick_channels(left_sensors)
# raw_right = raw.copy().pick_channels(right_sensors)

# %%
# Freq_Bands = {"theta": [4.0, 8.0], "alpha": [8.0, 13.0], "beta": [13.0, 30.0]}
Freq_Bands = {"beta": [13.0, 30.0]}

n_freq_bands = len(Freq_Bands)
min_freq = np.min(list(Freq_Bands.values()))
max_freq = np.max(list(Freq_Bands.values()))
freqs = np.linspace(min_freq, max_freq, int((max_freq - min_freq) * 4 + 1))

# # Define function for plotting con matrices
# def plot_con_matrix(con_data, n_con_methods):
#     """Visualize the connectivity matrix."""
#     fig, ax = plt.subplots(1, n_con_methods, figsize=(6 * n_con_methods, 6))
#     for c in range(n_con_methods):
#         # Plot with imshow
#         con_plot = ax[c].imshow(con_data[c, :, :, foi], cmap="binary", vmin=0, vmax=1)
#         # Set title
#         ax[c].set_title(connectivity_methods[c])
#         # Add colorbar
#         fig.colorbar(con_plot, ax=ax[c], shrink=0.7, label="Connectivity")
#         # Fix labels
#         ax[c].set_xticks(range(len(ch_names)))
#         ax[c].set_xticklabels(ch_names)
#         ax[c].set_yticks(range(len(ch_names)))
#         ax[c].set_yticklabels(ch_names)
#         print(
#             f"Connectivity method: {connectivity_methods[c]}\n"
#             + f"{con_data[c,:,:,foi]}"
#         )
#     return fig


con_epoch = mne_connectivity.spectral_connectivity_epochs(data=meglfp_epochs, fmin=min_freq, fmax=max_freq,
method='coh', mode='multitaper', sfreq=sfreq, faverage=True, n_jobs=-1)

mne_connectivity.viz.plot_connectivity_circle(con_epoch.get_data(output='dense').mean(2), meglfp_epochs.ch_names)

# %%
# tuple of lists
ragged_indices = ([[0, 1   ], [0, 1, 2, 3]],
                  [[2, 3, 4], [4         ]])

# tuple of tuples
# ragged_indices = (((0, 1   ), (0, 1, 2, 3)),
#                   ((2, 3, 4), (4         )))

# tuple of arrays
# ragged_indices = (np.array([[0, 1   ], [0, 1, 2, 3]], dtype='object'),
#                   np.array([[2, 3, 4], [4         ]], dtype='object'))

# ragged_indices = (np.array([[0, 0, 0], [1, 1, 1]], dtype='object'),
#                   np.array([[2, 3, 4], [2, 3, 4]], dtype='object'))

print(ragged_indices)

# ragged_indices = (np.array(seed.astype('object').tolist(), dtype='object'),
#            np.array(target.astype('object').tolist(), dtype='object'))


# create random data
data = np.random.randn(10, 5, 200)  # epochs x channels x times
sfreq = 50
ragged_indices = ([[0, 1], [0, 1, 2, 3]], [[2, 3, 4], [4]])  # seeds  # targets

# compute connectivity
con = mne_connectivity.spectral_connectivity_epochs(
    data,
    method="mic",
    indices=ragged_indices,
    sfreq=sfreq,
    fmin=10,
    fmax=30,
    verbose=False,
)

# print(ragged_indices)
# %%

# alex helper
arr_tremor_sensor = np.vstack((lfp_epochs._data, meg_epochs._data))
arr_tremor_sensor = meglfp_epochs._data
# arr_tremor_sensor = np.vstack((lfp._data, meg._data))


array = np.vstack((meg._data, lfp._data))
array = array[np.newaxis, :, :]

indices = (np.zeros((1,n_meg_sensors))[0].astype(int),
            np.arange(1,n_meg_sensors+1).reshape((1,n_meg_sensors))[0].astype(int))
 
n_meg_sensors =  len(meg.ch_names)
n_lfp_sensors = len(lfp.ch_names) 


lfp_indices = np.tile(np.arange(n_lfp_sensors).reshape((-1, 1)), reps=n_meg_sensors).astype(int)
meg_indices = np.tile(np.arange(n_lfp_sensors, n_meg_sensors + n_lfp_sensors), reps=n_lfp_sensors).reshape(n_lfp_sensors, n_meg_sensors).astype(int)

indices = (lfp_indices[0], meg_indices[0])

# indices = (np.tile(np.arange(n_lfp_sensors).reshape((-1, 1)), reps=n_meg_sensors).astype(int),
#        np.tile(np.arange(n_lfp_sensors, n_meg_sensors + n_lfp_sensors), reps=n_lfp_sensors).reshape(n_lfp_sensors, n_meg_sensors).astype(int))

# Convert lists to arrays
seeds = np.array(lfp_indices, dtype='object')
targets = np.array(meg_indices, dtype='object')

seeds = [lfp for lfp in lfp_indices]
targets = [meg for meg in meg_indices]
seeds = np.array(seeds, dtype='object')
targets = np.array(meg_indices, dtype='object')


# Create indices as a tuple of arrays
indices = (seeds, targets)

ind = mne_connectivity.seed_target_indices(lfp_indices, meg_list) 
ind = mne_connectivity.seed_target_indices_multivariate(lfp_indices, meg_list) 
indices = mne_connectivity.seed_target_indices([lfp_list], [meg_list])
indices = (lfp_indices[0], meg_indices[0])

indices = (lfp_list, meg_list)

# arr_tremor_sensor = arr_tremor_sensor[np.newaxis,:,:]
coh_lfp_meg_tremor2 = mne_connectivity.spectral_connectivity_time(array, freqs=freqs,method='coh',sfreq=sfreq,indices=indices, n_jobs=-1)

coh_lfp_meg_tremor = mne_connectivity.spectral_connectivity_epochs(arr_tremor_sensor, fmin=13, fmax=30, method='coh',sfreq=sfreq,indices=indices, n_jobs=-1)

arr_coh_lfp_meg_tremor = coh_lfp_meg_tremor2.get_data()
mean = np.mean(arr_coh_lfp_meg_tremor, axis=0) # average across all epochs
trem_coh_spec = mne.time_frequency.SpectrumArray(mean, meg_epochs.info, freqs=freqs)
trem_coh_spec = mne.time_frequency.SpectrumArray(arr_coh_lfp_meg_tremor, meg_epochs.info, freqs=np.array(coh_lfp_meg_tremor.freqs))


coh_lfpmeg_epoch = coh_lfp_meg_tremor._data.reshape(n_lfp_sensors, n_meg_sensors, len(coh_lfp_meg_tremor.freqs))
coh_lfpmeg_time = np.mean(coh_lfp_meg_tremor2._data, axis=0).reshape(n_lfp_sensors, n_meg_sensors, len(coh_lfp_meg_tremor2.freqs))


spectrum_epoch = mne.time_frequency.SpectrumArray(np.mean(coh_lfpmeg_epoch, axis=0), meg_epochs.info, freqs=np.array(coh_lfp_meg_tremor.freqs))
spectrum_time = mne.time_frequency.SpectrumArray(np.mean(coh_lfpmeg_time, axis=0), meg_epochs.info, freqs=np.array(coh_lfp_meg_tremor2.freqs))

# %%



# fig, ax = plt.subplots()
spectrum_epoch.plot_topomap(bands={'beta': (min(freqs), max(freqs))}, res=300, cmap=(parula_map, True), dB=True)
spectrum_time.plot_topomap(bands={'beta': (min(freqs), max(freqs))}, res=300, cmap=(parula_map, True), dB=True)

# ax[0].set_title('Epoch')
# ax[1].set_title('Time')
# plt.title('Beta band (13-30) Hz')
plt.show()



# con_epoch_right = mne_connectivity.spectral_connectivity_epochs(data=data_epochs_right, fmin=min_freq, fmax=max_freq,
# method='coh', mode='multitaper', sfreq=sfreq, faverage=True, n_jobs=-1)
# con_epoch_left = mne_connectivity.spectral_connectivity_epochs(data=data_epochs_left, fmin=13, fmax=30, method='coh', sfreq=sfreq, n_jobs=-1)

# con_time = mne_connectivity.spectral_connectivity_time(data=data_epochs, freqs=5, method='coh', sfreq=time_sample, n_jobs=-1)


# %%
con_right = con_epoch_right.get_data(output='dense')
con_r = np.mean(con_right, axis=-1)

con_left = con_epoch_left.get_data(output='dense')
con_l = np.mean(con_left, axis=-1)

# %%
# Plot the global connectivity over time
n_channels = meglfp_epochs.info["nchan"]  # get number of channels
times = meglfp_epochs.times[meglfp_epochs.times >= 1]  # get the timepoints
n_connections = (n_channels * n_channels - n_channels) / 2

# Get global avg connectivity over all connections
con_epochs_raveled_array = con_epoch_right.get_data(output="raveled")
global_con_epochs = np.sum(con_epochs_raveled_array, axis=0) / n_connections

# Since there is only one freq band, we choose the first dimension
global_con_epochs = global_con_epochs[0]

fig = plt.figure()
plt.plot(times, global_con_epochs)
plt.xlabel("Time (s)")
plt.ylabel("Global theta wPLI over trials")

# Get the timepoint with highest global connectivity right after stimulus
t_con_max = np.argmax(global_con_epochs[times <= 0.5])
print(f"Global theta wPLI peaks {times[t_con_max]:.3f}s after stimulus")
# %%
fig, ax = plt.subplots()
con_plot = ax.imshow(con_r[:, :, 0])


# %%
# meg_sides = meg_epochs.copy().rename_channels(mapping).ch_names
# meg_sides = meg_epochs.ch_names
# name_right = meg_sides + lfp_right_epochs.ch_names
# name_left = meg_sides + lfp_left_epochs.ch_names


# times = np.array([0.0])
freq = 20
coh_topo_r = con_right[:, :, freq]
coh_evok_r = mne.EvokedArray(con_r, info_r.info)
coh_evok_r.plot_topomap()

coh_topo_l = con_left[:, :, freq]
coh_evok_l = mne.EvokedArray(coh_topo_l, info_l.info)
coh_evok_l.plot_topomap()



# %%

ave_rest = epochs_rest.average()
ave_task = epochs_task.average()

evokeds = dict(rest=ave_rest, task=ave_task)
mne.viz.plot_compare_evokeds(evokeds, combine='mean')

# %%
# ave_rest = epochs_rest.average()
# ave_task = epochs_task.average()

evokeds = dict(rest=list(epochs_rest.iter_evoked()),
               task=list(epochs_task.iter_evoked()),)
mne.viz.plot_compare_evokeds(evokeds, combine='mean')

# %%
comb = mne.combine_evoked([ave_rest, ave_task], weights=[-1, 1])
comb.plot_joint()



# %%
fig, axes = plt.subplots(1, 2, facecolor='black', subplot_kw=dict(polar=True))
sides = ['left', 'right']

# meg_sensors = [for ch in meg_epochs.ch_names]
# mapping = dict(zip(left_sensors + right_sensors, ['LEFT'] * len(left_sensors) + ['RIGHT'] * len(right_sensors)))

mapping = {name: f'LEFT{i}' if name in left_sensors else f'RIGHT{i}' for i, name in enumerate (right_sensors + left_sensors, start=1)}
meg_sides = meg_epochs.copy().rename_channels(mapping).ch_names
name_right = meg_sides + lfp_right_epochs.ch_names
name_left = meg_sides + lfp_left_epochs.ch_names

for ax, side in zip(axes, sides):
    if side == 'left':
        mne_connectivity.viz.plot_connectivity_circle(con_l, name_left, title=side, ax=ax)
    else:
        mne_connectivity.viz.plot_connectivity_circle(con_r, name_right, title=side, ax=ax)



# info_con = mne.create_info(node_name, sfreq=downsample)
# mne_connectivity.viz.plot_sensors_connectivity(info_con, connectivity_mean)


# %%
freq_bin = 10
con_freqs = np.array(con_epoch.freqs)
freq_idx = np.argmin(np.abs(con_freqs - freq_bin))

# %%


data = np.mean(data, axis=2)
# Create SpectralConnectivity object
n_epochs = len(meg_epochs)
n_nodes_meg = len(meg_epochs.ch_names)
n_nodes_lfp = len(lfp_epochs.ch_names)
n_nodes_total = n_nodes_meg + n_nodes_lfp
# n_freqs = 3  # Example frequency range

data = np.vstack((meg_data_epochs, lfp_data_epochs))
data = data[np.newaxis, :, :]

# %%
array = np.vstack((meg._data, lfp._data))
array = array[np.newaxis, :, :]

meglfp_coh = mne_connectivity.spectral_connectivity_time(array, freqs=freqs,
             method='coh', sfreq=sfreq, n_jobs=-1, average=True)
# meglfp_coh = mne_connectivity.SpectralConnectivity(array, freqs, 1)




connectivity = mne_connectivity.SpectralConnectivity(data=data, freqs=freqs, n_nodes=n_nodes_total, method='coh')

connectivity = mne_connectivity.SpectralConnectivity(data=data, method='coh', freqs=freqs, n_nodes=n_nodes_total)

# %%
# get the events from the raw annotations
events, events_id = mne.events_from_annotations(raw_segments)

# make fixed_events assuming that there is one move every 2 seconds with an overlap of 1 sec
events = mne.make_fixed_length_events(raw_segments, duration=2, overlap=1)
events_id = {task:1}

# epochs the data and reject the bad ones by annotation
epochs = mne.Epochs(raw_segments, events, events_id, preload=True, reject_by_annotation=True)

# %%
# standard tfr 
frequencies = np.arange(7, 30, 3)
power = mne.time_frequency.tfr_morlet(
    epochs, n_cycles=2, return_itc=False, freqs=frequencies, decim=3, n_jobs=-1)
power.plot(['LFP-right-0'])


#%%
emgs = raw.get_data()
sfreq = raw.info['sfreq']

# %%
ch_names = raw.ch_names

emg_right = emgs[:2]
emg_left = emgs[2:]

mean_right = np.mean(emg_right)
mean_left = np.mean(emg_left)

rms_right = np.sqrt(np.mean(emg_right**2))
rms_left = np.sqrt(np.mean(emg_left**2))

# selected_emg =  emg_left if mean_left > mean_right else  emg_right  
# selected_emg_label = 'Left' if mean_left > mean_right else 'Right' 
# print(selected_emg_label)

selected_emg =  emg_left if rms_left < rms_right else  emg_right  
selected_emg_label = 'Left' if rms_left < rms_right else 'Right' 

if selected_emg_label == 'Right':

    ch_names = ch_names[:2]
else:
    ch_names = ch_names[2:]
    
print(selected_emg_label, ch_names)


# need to selected appropriate timing of the raw before computing the EMG


# %%
# take the mean of both channels 
selected_emg = np.mean(selected_emg, axis=0)

# %%
# selected_emg = raw.copy().pick_channels(ch_names)

# spectrum2 = selected_emg.compute_psd(method='welch', fmin=0, n_jobs=-1)
#%%

psd, freqs = mne.time_frequency.psd_array_welch(selected_emg, sfreq=sfreq, fmin=0, fmax=400, n_fft=2048)
# plt.figure()
plt.plot(freqs, psd)
# mean_psd = psd.mean(0)
# mean_psd = mean_psd.reshape(1, 205)

# %%
# info = mne.create_info(ch_names, sfreq=sfreq, ch_types='eeg') # pass it as EEG to create the info even though it is EMG here 
info = mne.create_info(['EMG'], sfreq=sfreq, ch_types='eeg') # pass it as EEG to create the info even though it is EMG here 

# spectrum = mne.time_frequency.SpectrumArray(data=psd, freqs=freqs, info=info)
spectrum = mne.time_frequency.Spectrum(selected_emg_psd, method='welch', fmin=0, fmax=200)


# %%
spectrum.plot()
