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

from utils import get_raw_condition, get_emg_power, get_selected_emg, plot_power
# %matplotlib qt 

# %%
mapping = {  'S001':'sub-hnetKS',
             'S002':'sub-QZTsn6',
             'S003':'sub-VopvKx',
             'S004':'sub-jyC0j3',
             'S005':'sub-AB2PeX',
             'S006':'sub-gNX5yb',
             'S007':'sub-BgojEx',
             'S008':'sub-BYJoWR',
             'S009':'sub-0cGdk9',
             'S010':'sub-2IhVOz',
             'S011':'sub-FYbcap',
             'S012':'sub-PuPVlx',
             'S013':'sub-zxEhes',
             'S014':'sub-2IU8mi',
             'S015':'sub-i4oK0F',
             'S016':'sub-FIyfdR',
             'S017':'sub-dCsWjQ',
             'S018':'sub-iDpl28',
             'S019':'sub-AbzsOg',
             'S020':'sub-oLNpHd',
             'S021':'sub-8RgPiG',
             'S022':'sub-6m9kB5'}

rhm_mapping = {'rhm{:03d}'.format(int(key[1:])): value for key, value in mapping.items()}
rev_rhm_mapping = {v:k for k, v in rhm_mapping.items()}
# dict(sorted(rev_rhm_mapping.items(), key=lambda item:item[0])) #reverse order
rhm_mapping2 = {subject:key.replace('sub-', '')for subject, key in rhm_mapping.items()}


bids_root = '/data/raw/hirsch/RestHoldMove/'
anon_root = '/data/raw/hirsch/RestHoldMove_anon'

check_side = []

fmax=45
baseline = ''
all_ground_subject = []

for i in range(1, 23):
    n = str(i).zfill(3)
    subject = f'rhm{n}'
    anon_subject = rhm_mapping2[subject]
    
    # get every subject bids information in the dataset 
    # subject_files = BIDSPath(root=bids_root, subject=subject, extension='fif').match()
    
    subject_files = BIDSPath(root=anon_root, session='PeriOp',  subject=anon_subject, extension='.fif').match()
    subject_files = [file for file in subject_files if 'noise' not in file.basename]
    subject_files = [file for file in subject_files if 'split-02' not in file.basename]
    
    ground_subject = [] # all psd grounds for one subject
    
    # total_sample = []
    rest_sample = []
    hold_sample = []
    move_sample = []

    # copy the subject_files inside the anonymized folder with the new filenaming scheme
    for bids in subject_files:
        raw = read_raw_bids(bids)
                    
        # pick all emg channels
        channels = ['EMG061', 'EMG062', 'EMG063', 'EMG064']
        raw.pick_channels(channels)
        
        # load the data
        raw.load_data()
        
        # apply a 1Hz high pass firwin filter to remove slow drifts
        raw.filter(l_freq=1, h_freq=None, method='fir', n_jobs=-1, picks=['emg'])
        
        # apply a notch filter
        # raw.notch_filter(freqs)
        
        if bids.task == 'Rest': # true only for last 2 subject
            conditions = ['rest'] 
        else:
            conditions = ['rest'] + [bids.task]
        
        
        rest_segments, task_segments = get_raw_condition(raw, conditions)
        
        if i <  21: # last 2 subjects only had resting state
            # task_segments.plot(duration=20) # quickly plot to check the correct side
            psd_task, freqs, side, ch = get_emg_power(task_segments, raw.ch_names, side='find', fmax=fmax)
            if 'Hold' in bids.task:
                hold_sample.append(len(task_segments.times))
            elif 'Move' in bids.task:
                move_sample.append(len(task_segments.times))
                
        else:
            psd_task = None
            
        if rest_segments is not None:
            if i < 21:
                psd_rest, _, _, _= get_emg_power(rest_segments, raw.ch_names, side=side, fmax=fmax)

            else:
                psd_rest, freqs, side, ch = get_emg_power(rest_segments, raw.ch_names, side='all', fmax=fmax) # average overall EMGs
        
            rest_sample.append(len(rest_segments))

        else:
            psd_rest = None
        
        
        if baseline != 'rest':
            if rest_segments is not None and task_segments is not None:
                ground = mne.io.concatenate_raws([rest_segments.copy(), task_segments.copy()])
            elif rest_segments is None and task_segments is not None:
                ground = mne.io.concatenate_raws([task_segments.copy()])
            elif rest_segments is not None and task_segments is None:
                ground = mne.io.concatenate_raws([rest_segments.copy()])
        elif baseline == 'rest':
            if rest_segments is not None:
                ground = mne.io.concatenate_raws([rest_segments.copy()])
    
        # get ground power average
        psd_ground, _, _, _ = get_emg_power(ground, raw.ch_names, side=side, fmax=fmax)
        ground_subject.append(psd_ground)
        
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


        ground_basename = bids.copy().update(check=False, suffix=f'ground{baseline}', extension='.npy', split=None,
                                             acquisition=None, task=None, run=None).basename

        sub_path = Path(f'emgs/{bids.subject}/')
        sub_path.mkdir(parents=True, exist_ok=True)        
        emg_path = Path(f'{sub_path}/{basename}')
        emg_ground_path = Path(f'{sub_path}/{ground_basename}')
        
        if psd_rest is not None and psd_task is not None :
            np.savez(emg_path, freqs=freqs, psd_rest=psd_rest, psd_task=psd_task, task_time=task_segments.times, rest_time=rest_segments.times)
    
        elif psd_rest is None and psd_task is not None :
            np.savez(emg_path, freqs=freqs, psd_task=psd_task, task_time=task_segments.times)
        
        elif psd_rest is not None and psd_task is None :
            np.savez(emg_path, freqs=freqs, psd_rest=psd_rest, rest_time=rest_segments.times)

        # clear every variable that are saved
        del psd_rest, psd_task, freqs


        # append the filename, the auto found side and the actual task name for subject < 21
        if i < 21:
            check_side.append([emg_path, side, bids.task])
        
        # break
    np.save(emg_ground_path, np.mean(ground_subject, axis=0))
    
    weight_basename = bids.copy().update(check=False, suffix='weight', extension='.npy', split=None,
                                         acquisition=None, task=None, run=None)
    
    if sum(rest_sample) > 0 :
        weight_path = weight_basename.copy().update(task='rest').basename
        weight_path = Path(f'{sub_path}/{weight_path}')
        np.save(weight_path, sum(rest_sample))
    
    if sum(hold_sample) > 0 :
        weight_path = weight_basename.copy().update(task='hold').basename
        weight_path = Path(f'{sub_path}/{weight_path}')
        np.save(weight_path, sum(hold_sample))
    
    if sum(move_sample) > 0 :
        weight_path = weight_basename.copy().update(task='move').basename
        weight_path = Path(f'{sub_path}/{weight_path}')        
        np.save(weight_path, sum(move_sample))
    # break 
# %%
df = pd.DataFrame(check_side, columns=['file', 'side', 'task'])
df['check_side'] = 'left' if (df['task'] == 'HoldL') or (df['task'] == 'MoveL') else 'right'
df['check_side'] = np.where(df['task'].isin(['HoldL', 'MoveL']), 'left', 'right')
mismatch = df[df['side'] != df['check_side']]
# %%


emg_load = np.load(emg_path)

get_raw_condition(raw, conditions)
