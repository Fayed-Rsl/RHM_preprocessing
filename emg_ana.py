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
%matplotlib qt 

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

baseline = ''
all_psds_rest = []
all_psds_hold = []
all_psds_move = []

all_ave_psds_rest = []
all_ave_psds_hold = []
all_ave_psds_move = []

for i in range(1, 23):
    n = str(i).zfill(3)
    subject = f'rhm{n}'
    anon_subject = rhm_mapping2[subject]
    
    # get every subject bids information in the dataset 
    # subject_files = BIDSPath(root=bids_root, subject=subject, extension='fif').match()
    
    subject_files = BIDSPath(root=anon_root, session='PeriOp',  subject=anon_subject, extension='.fif').match()
    subject_files = [file for file in subject_files if 'noise' not in file.basename]
    subject_files = [file for file in subject_files if 'split-02' not in file.basename]
    
    # if i == 21:
    #     break 

    weight_basename = subject_files[0].copy().update(check=False, suffix='weight', extension='.npy', split=None,
                                         acquisition=None, task=None, run=None)
    
    sub_path = Path(f'emgs/{anon_subject}/')
    
    weight_rests = []
    weight_holds = []
    weight_moves = []
    
    psd_rests = []
    psd_holds = []
    psd_moves = []
    
    # copy the subject_files inside the anonymized folder with the new filenaming scheme
    for bids in subject_files: 
        # save the power for further group analysis
        # save the power for further group analysis
        basename = bids.copy().update(check=False, suffix='emgdata', extension='.npz', split=None).basename
        ground_basename = bids.copy().update(check=False, suffix=f'ground{baseline}', extension='.npy', split=None,
                                             acquisition=None, task=None, run=None).basename



    
        weight_path = weight_basename.copy().update(task='rest').basename
        weight_path = Path(f'{sub_path}/{weight_path}')
    


                
    
        sub_path = Path(f'emgs/{bids.subject}/')
        emg_path = Path(f'{sub_path}/{basename}')
        emg_ground_path = Path(f'{sub_path}/{ground_basename}')

        psd_ground = np.load(emg_ground_path)
        emg_load = np.load(emg_path)
        
        # load the freqs only once
        if i == 1:
            freqs = emg_load['freqs']

        try: # because some files does not have rest 
            psd_rest = emg_load['psd_rest']
            rest_time = emg_load['rest_time']
            # psd_rest = (psd_rest - psd_ground) / psd_ground  
            # psd_rest = ((psd_rest - psd_ground) / sum(psd_ground)) * 100
            
            psd_rest = ((psd_rest - psd_ground) / psd_ground) * 100
            all_psds_rest.append(psd_rest)
                
            weight_path = weight_basename.copy().update(task='rest').basename
            weight_path = Path(f'{sub_path}/{weight_path}')
            weight = np.load(weight_path)
            psd_rests.append(psd_rest)
            weight_rests.append(len(rest_time) / weight)
            
        except:
            pass # just skip if there is no rest


        if i < 21:
            psd_task = emg_load['psd_task']
            psd_task = ((psd_task - psd_ground) / psd_ground) * 100
            # psd_task = (psd_task - psd_ground) / psd_ground # normalize
            
        if 'Hold' in bids.task:
            weight_path = weight_basename.copy().update(task='hold').basename
            weight_path = Path(f'{sub_path}/{weight_path}')
            all_psds_hold.append(psd_task)
            hold_time = emg_load['task_time']
            
            weight = np.load(weight_path)
            psd_holds.append(psd_task)
            weight_holds.append(len(hold_time) / weight)
            
        elif 'Move' in bids.task:
            weight_path = weight_basename.copy().update(task='move').basename
            weight_path = Path(f'{sub_path}/{weight_path}')              
            all_psds_move.append(psd_task)
            move_time = emg_load['task_time']
            
            weight = np.load(weight_path)
            psd_moves.append(psd_task)
            weight_moves.append(len(move_time) / weight)
            
        # break
    weighted_average_rest = np.average(psd_rests, axis=0, weights=weight_rests)
    if i < 21:
        weighted_average_hold = np.average(psd_holds, axis=0, weights=weight_holds)
        weighted_average_move = np.average(psd_moves, axis=0, weights=weight_moves)
    
    all_ave_psds_rest.append(weighted_average_rest)
    if i < 21:
        all_ave_psds_hold.append(weighted_average_hold)
        all_ave_psds_move.append(weighted_average_move)
    # break

        
# %%

total_rest = len(all_psds_rest)
total_hold = len(all_psds_hold)
total_move = len(all_psds_move)
print(total_rest, total_hold, total_move)


# %%
total_rest = len(all_ave_psds_rest)
total_hold = len(all_ave_psds_hold)
total_move = len(all_ave_psds_move)
print(total_rest, total_hold, total_move)

mean_rest = np.mean(np.vstack(all_ave_psds_rest), axis=0)
mean_hold = np.mean(np.vstack(all_ave_psds_hold), axis=0)
mean_move = np.mean(np.vstack(all_ave_psds_move), axis=0)

std_rest = np.std(np.vstack(all_ave_psds_rest), axis=0)
std_hold = np.std(np.vstack(all_ave_psds_hold), axis=0)
std_move = np.std(np.vstack(all_ave_psds_move), axis=0)
#
# %%

mean_rest = np.mean(np.vstack(all_psds_rest), axis=0)
mean_hold = np.mean(np.vstack(all_psds_hold), axis=0)
mean_move = np.mean(np.vstack(all_psds_move), axis=0)

std_rest = np.std(np.vstack(all_psds_rest), axis=0)
std_hold = np.std(np.vstack(all_psds_hold), axis=0)
std_move = np.std(np.vstack(all_psds_move), axis=0)
# %%
plt.figure()
plt.plot(freqs, mean_rest, label='Rest', alpha=0.7)
plt.fill_between(freqs, mean_rest - std_rest, mean_rest + std_rest, alpha=0.2)

plt.plot(freqs, mean_hold, label='Hold', alpha=0.7)
plt.fill_between(freqs, mean_hold - std_hold, mean_hold + std_hold, alpha=0.2)


# plt.plot(freqs, mean_move, label='Move', alpha=0.7)
plt.fill_between(freqs, mean_move - std_move, mean_move + std_move, alpha=0.2)


plt.legend()
plt.show()