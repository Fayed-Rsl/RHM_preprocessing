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
from utils import parula_map, get_raw_condition

# %matplotlib qt 

# %%
bids_root = '/data/raw/hirsch/RestHoldMove_anon/'
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

# create a dict of all wanted channel types
channel_to_pick = {'meg':'grad', 'eeg': True} 

# pick all channels types and exclude the bads one
raw.pick_types(**channel_to_pick, exclude='bads') # if there is any bad channel remove them

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

# load the data
raw.load_data()

# set eeg reference using average scheme
raw.set_eeg_reference(ref_channels='average')

# apply a 1Hz high pass firwin filter to remove slow drifts
raw.filter(l_freq=1, h_freq=None, method='fir', n_jobs=-1)

# crop the data and get the rest_segments only
raw, _ = get_raw_condition(raw, ['rest'])

# downsample to 200 from 2000 
raw.resample(200) 

# get the new sampling freq
sfreq = raw.info['sfreq']

# freq up to beta (13, 30)
freqs = np.arange(13, 31, 1)
fmin = min(freqs)
fmax = max(freqs)
freqs_beta = {'beta': (fmin, fmax)}

# create epochs of fixed length 2 seconds and 50% overlap.
epochs = mne.make_fixed_length_epochs(raw, duration=2, overlap=1, preload=True)

# get the epochs data to compute connectivity in
data = epochs.get_data()

# del the raw object since we will work with epochs from now on
del raw
# %%
# get the LFP channel names for right and left side
ch_lfp = epochs.copy().pick_types(eeg=True).ch_names
ch_lfp_right = [ch for ch in ch_lfp if 'right' in ch]
ch_lfp_left = [ch for ch in ch_lfp if 'left' in ch]

# find the indices of the desired channel names
lfp_right_ind = [ch_lfp.index(name) for name in ch_lfp_right]
lfp_left_ind = [ch_lfp.index(name) for name in ch_lfp_left]
same_size = len(lfp_right_ind) == len(lfp_left_ind)
print(f'LFP right and left are equal: {same_size}')

# get the position of the sensor to split between ch meg left
meg = epochs.copy().pick_types(meg='grad')
lfp = epochs.copy().pick_types(eeg=True)
sensor_pos = meg.info['chs']
ch_meg_left = [ch_name for i, ch_name in enumerate(meg.ch_names) if sensor_pos[i]['loc'][0] < 0]
ch_meg_right = [ch_name for i, ch_name in enumerate(meg.ch_names) if sensor_pos[i]['loc'][0] > 0]

# get the channel length for meg and lfp 
n_meg_sensors =  len(meg.ch_names)
n_lfp_sensors = len(lfp.ch_names) 

# create indices: seeds and targets for connectivity measure so : get the index where it is LFP and MEG
seeds  = mne.pick_types(epochs.info, eeg=True)# going from 0 to 202 MEG sensors as it in the epochs._data 
targets = mne.pick_types(epochs.info, meg=True) # we want the connectivity between all sensors
# n_connection indices[0] should be equal to n_lfp_sensors * n_meg_sensors
indices = mne_connectivity.seed_target_indices(seeds, targets) 

# %%
coh_lfpmeg = mne_connectivity.spectral_connectivity_epochs(data, fmin=fmin, fmax=fmax, method='coh', mode='multitaper',
                                                           sfreq=sfreq, indices=indices, n_jobs=-1)


coh_lfpmeg.save('coherence/test2')

# %%
test = mne_connectivity.read_connectivity('coherence/test2')

# %%
# get the data of the connectivity coherence --> shape should be (N_connection * N_freqs)
coh_data = coh_lfpmeg.get_data() 
coh_freqs = np.array(coh_lfpmeg.freqs)
# reshape the data according to (N_lfp * N_meg * N_freqs)
coh_data = coh_data.reshape(n_lfp_sensors, n_meg_sensors, len(coh_freqs))

# average the connectivity on LFP axis channel, average type: either all LFP, only left or right side
# average_type = 'all'
coh_average = np.mean(coh_data, axis=0)
coh_average_left = np.mean(coh_data[lfp_left_ind, :, :], axis=0) # get only the LFP left
coh_average_right = np.mean(coh_data[lfp_right_ind, :, :], axis=0) # get only the LFP right

# create a mne Spectrum array using MEG sensors information
coh_spec = mne.time_frequency.SpectrumArray(coh_average, meg.info, freqs=coh_freqs)
coh_spec_left = mne.time_frequency.SpectrumArray(coh_average_left, meg.info, freqs=coh_freqs)
coh_spec_right = mne.time_frequency.SpectrumArray(coh_average_right, meg.info, freqs=coh_freqs)
# coh_spec_diff = mne.time_frequency.SpectrumArray((coh_average_right - coh_average_left), meg.info, freqs=coh_freqs) 

# %%

fig, ax = plt.subplots(1, 3, figsize=(10, 8))

# fieldtrip colors
coh_spec.plot_topomap(bands=freqs_beta, res=300, cmap=(parula_map, True), dB=False, axes=ax[0])
coh_spec_left.plot_topomap(bands=freqs_beta, res=300, cmap=(parula_map, True), dB=False, axes=ax[1])
coh_spec_right.plot_topomap(bands=freqs_beta, res=300, cmap=(parula_map, True), dB=False, axes=ax[2])
# coh_spec_diff.plot_topomap(bands=freqs_beta, res=300, cmap=(parula_map, True), dB=True, axes=ax[3])

# mne colors 
# coh_spec.plot_topomap(bands=freqs_beta, res=300, dB=True, axes=ax[0])
# coh_spec_left.plot_topomap(bands=freqs_beta, res=300,  dB=True, axes=ax[1])
# coh_spec_right.plot_topomap(bands=freqs_beta, res=300, dB=True, axes=ax[2])

ax[0].set_title('All LFP')
ax[1].set_title('LFP left')
ax[2].set_title('LFP right')
# ax[3].set_title('LFP L-R')


plt.show()
