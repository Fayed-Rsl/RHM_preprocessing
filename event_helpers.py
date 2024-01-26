
# %%
# import the main libraries required for the preprocessing
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree
import mne 
import pandas as pd 
from pathlib import Path
from scipy.io import loadmat
import numpy as np
import neurokit2 as nk
from utils import emg_event, plot_emg_burst
import matplotlib.pyplot as plt

# %matplotlib qt 
# %%
bids_root = '/data/raw/hirsch/RestHoldMove_anon2/'
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
# read the raw data
raw = read_raw_bids(bids_path)
# %%

emg_info = emg_event(raw, time_for_burst=10, end_rest=300, factor=3, emg_ch=['EMG063', 'EMG064'], split_file=False)
timestamps = emg_info['timestamps']
tkeo, threshold = emg_info['tkeo'], emg_info['threshold']
bursting, burst_duration = emg_info['bursting'], emg_info['burst_duration']
onsets, offsets = emg_info['onsets'], emg_info['offsets']
print(onsets)
onsets_idx, offsets_idx = emg_info['onsets_idx'], emg_info['offsets_idx']

# onsets_idx = [np.where(timestamps == onset)[0] for onset in onsets]

# %%
plot_emg_burst(timestamps, tkeo, bursting, onsets, offsets, threshold)  

# %%


# plt.plot(timestamps[1:], tkeo, alpha=0.5, label='TKEO')
# plt.plot(timestamps[1:], bursting, alpha=0.5, label='Detected Burst')
# for i, line in enumerate(burst_periods[:-1]):
for i, (on, off) in enumerate(zip(onsets, offsets)):
    # only write the label once
    plt.axvline(on, color='green', linestyle='--', linewidth=2, label='Onset' if i == 0 else None)
    plt.axvline(off, color='red', linestyle='--', linewidth=2, label='Offset' if i == 0 else None)
    # plt.fill_betweenx([0, max(tkeo)], on, off, color='purple', alpha=0.2, label='Burst Period' if i == 0 else None)
        
activity_signal = pd.Series(np.full(len(bursting), np.nan))
activity_signal[onsets_idx] = bursting[onsets_idx]
activity_signal[offsets_idx] = bursting[offsets_idx]
activity_signal = activity_signal.bfill()
if np.any(activity_signal.isna()):
    index = np.min(np.where(activity_signal.isna())) - 1
    value_to_fill = activity_signal[index]
    activity_signal = activity_signal.fillna(threshold)
    
# Plot Amplitude.
plt.plot(
    timestamps[1:],
    bursting,
    color="#FF9800",
    label="Amplitude",
    linewidth=1.5,
)

# Shade activity regions.
plt.fill_between(
    timestamps[1:],
    bursting,
    activity_signal,
    where=bursting > activity_signal,
    color="orange",
    alpha=0.5,
    label=None,
)

# %%
# def _emg_plot_activity(emg_signals, onsets, offsets):
#     activity_signal = pd.Series(np.full(len(emg_signals), np.nan))
#     activity_signal[onsets] = emg_signals["EMG_Amplitude"][onsets].values
#     activity_signal[offsets] = emg_signals["EMG_Amplitude"][offsets].values
#     activity_signal = activity_signal.bfill()

#     if np.any(activity_signal.isna()):
#         index = np.min(np.where(activity_signal.isna())) - 1
#         value_to_fill = activity_signal[index]
#         activity_signal = activity_signal.fillna(value_to_fill)

#     return activity_signal


# # Shade activity regions.
# activity_signal = _emg_plot_activity(emg_signals, onsets, offsets)
# ax1.fill_between(
#     x_axis,
#     emg_signals["EMG_Amplitude"],
#     activity_signal,
#     where=emg_signals["EMG_Amplitude"] > activity_signal,
#     color="#f7c568",
#     alpha=0.5,
#     label=None,
# )



activity_signal = pd.Series(np.full(len(emg_info['emg']), np.nan))
activity_signal[onsets_idx] = emg_info["tkeo"][onsets_idx]
activity_signal[offsets_idx] = emg_info["tkeo"][offsets_idx]
activity_signal = activity_signal.bfill()
if np.any(activity_signal.isna()):
    index = np.min(np.where(activity_signal.isna())) - 1
    value_to_fill = activity_signal[index]
    activity_signal = activity_signal.fillna(value_to_fill)

plt.fill_between(
    timestamps[1:],
    bursting,
    activity_signal[1:],
    where=bursting,
    color="#f7c568",
    alpha=1,
    label=None,
)


for i, (onset, offset) in enumerate(zip(onsets, offsets)):
    plt.fill_between(timestamps[1:], 0, max(tkeo),
                      where=(timestamps[1:] >= onset) & (timestamps[1:] <= offset))

    plt.fill(timestamps[1:], 0, max(tkeo),
                      where=(timestamps[1:] >= onset) & (timestamps[1:] <= offset))

                  # on, off, color='purple', alpha=0.2, label='Burst Period' if i == 0 else None)

mask = (timestamps >= onset) & (timestamps <= offset)
val = timestamps[mask]
plt.fill_between(val, bursting[mask[1:]], color='skyblue', alpha=0.4)
plt.plot(val, bursting[mask[1:]], color='blue', alpha=0.6)

plt.fill_betweenx([0, max(tkeo)], on, off, color='purple', alpha=0.2, label='Burst Period' if i == 0 else None)

plt.axhline(y=threshold, color='black', linestyle='--', label='threshold')
plt.legend()
plt.show()

# %%

emg_signals = emg_info.copy()
x_axis = timestamps
x_axis_burst = timestamps[1:]

fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
fig.suptitle("Electromyography (EMG)", fontweight="bold")
plt.tight_layout(h_pad=0.2)

# Plot cleaned and raw EMG.
ax0.set_title("Raw and Cleaned Signal")
ax0.plot(x_axis, emg_signals["emg"], color="#B0BEC5", label="Raw", zorder=1)
ax0.legend(loc="upper right")


# Clean signal
emg_cleaned = nk.emg_clean(
    emg_info['emg'], sampling_rate=1000)

# Get amplitude
amplitude = nk.emg_amplitude(emg_cleaned)

# Plot Amplitude.
ax1.set_title("Muscle Activation")
ax1.plot(
    x_axis,
    amplitude,
    color="#FF9800",
    label="Amplitude",
    linewidth=1.5,
)

# def _emg_plot_activity(emg_signals, onsets, offsets):
    
activity_signal = pd.Series(np.full(len(amplitude), np.nan))
activity_signal[onsets_idx] = amplitude[onsets_idx]
activity_signal[offsets_idx] = amplitude[offsets_idx]
activity_signal = activity_signal.bfill()

if np.any(activity_signal.isna()):
    index = np.min(np.where(activity_signal.isna())) - 1
    value_to_fill = activity_signal[index]
    activity_signal = activity_signal.fillna(value_to_fill)

# Shade activity regions.
ax1.fill_between(
    x_axis,
    amplitude,
    activity_signal,
    where=amplitude > activity_signal,
    color="orange",
    alpha=0.5,
    label=None,
)

# %%

activity = np.array([])
for x, y in zip(onsets, offsets):
    activated = np.arange(x, y)
    activity = np.append(activity, activated)



# %%


# %%
sample=250
emg_ch = ['EMG063', 'EMG064']
# raw2 = raw.copy().pick_types(emg=True).pick_channels(emg_ch)  # default EMG 63-64
raw2 = raw.copy().pick_types(emg=True).pick_channels(emg_ch).resample(sample)  # default EMG 63-64
raw2.load_data()

# %%

t = raw2.times
no_rest = np.where(t >= 300)[0]
timestamps = t[no_rest]

emg = np.mean(raw2._data[:, no_rest], axis=0) # average emg channel
signals, info = nk.emg_process(emg, sampling_rate=1000)

# set the sampling_rate to the downsampled after processing to have matching time
info['sampling_rate'] = sample 
nk.emg_plot(signals, info)





# %%


# get the time and sampling freq
fs = raw.info['sfreq']
t = raw.times

if split_file == False: 
    # get the period when the rest is over
    no_rest = np.where(t >= end_rest)[0]
    timestamps = t[no_rest]
    emg = np.mean(raw._data[:, no_rest], axis=0) # average emg channel
else:
    timestamps = t[:]
    emg = np.mean(raw._data[:, :], axis=0) 

