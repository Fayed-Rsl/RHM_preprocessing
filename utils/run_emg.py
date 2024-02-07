# import the main libraries required for the preprocessing
from mne_bids import read_raw_bids
import mne 
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
# import homemade function 
from technical_validation_utils import get_raw_condition, get_emg_power, plot_power, subject_files
mne.set_log_level(verbose='CRITICAL') # reduce verbose output
# %%
# apply psd welch between 2, 45 Hz
fmin = 2
fmax = 45 

# loop over all subject files
for n_file, bids in enumerate(subject_files):

    # set the side of the EMG according to the task label
    if bids.task in ['MoveL', 'HoldL']:
        side = 'left'
    
    elif bids.task in ['MoveR', 'HoldR']:
        side = 'right'
    
    elif bids.task == 'Rest':
        side = 'all'
        
    # read the raw file
    raw = read_raw_bids(bids)
    
    # pick all emg channels
    channels = ['EMG061', 'EMG062', 'EMG063', 'EMG064']
    raw.pick_channels(channels)
    
    # load the data
    raw.load_data()
    
    # apply a 1Hz high pass firwin filter to remove slow drifts.
    raw.filter(l_freq=1, h_freq=None, method='fir', n_jobs=-1, picks=['emg'])
        
    # if the task is only Rest then keep the raw segments as it is for rest and set None to task_segments
    if bids.task == 'Rest': # true only for 2 subject
        rest_segments, task_segments = raw, None
    else:
        conditions = ['rest'] + [bids.task] # rest + task (hold or move)
        rest_segments, task_segments = get_raw_condition(raw, conditions)
    
    # compute psd for the task_segments in the side where the subject made the task
    if task_segments is not None: # (only when bids.task is not Rest)
        # comupute power of EMG in the side that the participant moved his hand
        psd_task, freqs, _, ch = get_emg_power(task_segments, raw.ch_names, side=side, fmin=fmin, fmax=fmax)

    else:
        psd_task = None
        
    # compute psd for the rest segments
    if rest_segments is not None:
        if bids.task != 'Rest':
            # if the task is not Rest then get power over the side were the task was effectuated 
            psd_rest, _, _, _= get_emg_power(rest_segments, raw.ch_names, side=side, fmin=fmin, fmax=fmax)

        elif bids.task == 'Rest':
            # if the task is Rest then get power on the averaged 4 EMGs 
            psd_rest, freqs, _, ch = get_emg_power(rest_segments, raw.ch_names, side=side, fmin=fmin, fmax=fmax)
        
    else:
        psd_rest = None
    
    # compute the psd ground (which is the combination of raw rest_segments + task segments time series)
    if rest_segments is not None and task_segments is not None:
        ground = mne.io.concatenate_raws([rest_segments.copy(), task_segments.copy()])
    elif rest_segments is None and task_segments is not None:
        ground = mne.io.concatenate_raws([task_segments.copy()])
    elif rest_segments is not None and task_segments is None:
        ground = mne.io.concatenate_raws([rest_segments.copy()])

    # get ground power average
    psd_ground, _, _, _ = get_emg_power(ground, raw.ch_names, side=side, fmax=fmax)
    
    # plot power spectrum 
    fig, ax = plt.subplots()
    if psd_task is not None:
        plot_power(psd_task, freqs, side, condition=bids.task, color='blue', ax=ax)
    if psd_rest is not None:
        plot_power(psd_rest, freqs, side, condition='rest', color='orange', ax=ax)
        
    plt.title(f'{ch} | sub {bids.subject} | {bids.acquisition}')
    plt.show()
                
    # save the power for further group analysis
    basename = bids.copy().update(check=False, suffix='emgdata', extension='.npz', split=None).basename
    sub_path = Path(f'emgs/sub-{bids.subject}/')
    sub_path.mkdir(parents=True, exist_ok=True)        
    emg_path = Path(f'{sub_path}/{basename}')
        
    if psd_rest is not None and psd_task is not None :
        np.savez(emg_path, freqs=freqs, psd_ground=psd_ground, psd_rest=psd_rest, psd_task=psd_task, task_time=task_segments.times, rest_time=rest_segments.times)

    elif psd_rest is None and psd_task is not None :
        np.savez(emg_path, freqs=freqs, psd_ground=psd_ground, psd_task=psd_task, task_time=task_segments.times)
    
    elif psd_rest is not None and psd_task is None :
        np.savez(emg_path, freqs=freqs, psd_ground=psd_ground, psd_rest=psd_rest, rest_time=rest_segments.times)

    # clear every variable that are saved
    del psd_rest, psd_task, psd_ground, freqs

