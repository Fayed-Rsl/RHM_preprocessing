# import the main libraries required for the preprocessing
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os 

# import homemade function 
from technical_validation_utils import subjects
    
# %%
# NB: run_emg.py must has been runned before running the average
emg_dir = [Path('./emgs/') / sub for sub in subjects]
assert all(file.exists() for file in emg_dir), 'Please run_emg.py. Path has not been created yet.'

# store variable for coherences 
all_ave_psds_rest = []
all_ave_psds_hold = []
all_ave_psds_move = []

norm = False # for load zscore on time series

for n_sub, subject_dir in enumerate(emg_dir):
    emg_files = [subject_dir / file for file in os.listdir(subject_dir) if 'emgdata' in file]
    if norm:
        emg_files = [file for file in emg_files if 'zscore' in file.__str__()] # zscore on time series --> then compute psd
    else:
        emg_files = [file for file in emg_files if 'zscore' not in file.__str__()] # time series --> then compute psd
    emg_files.sort() 

    # store all psds for one subjects to average over
    psd_rests = []
    psd_holds = []
    psd_moves = []
    
    # save the total time (sample) for each condition, in order to make a weighted average for each condition
    # i.e., the weight will be actual_condition_sample / total_condition_sample
    # (for example if a file is 80% it will have more weight in the average)
    
    rest_sample = []
    hold_sample = []
    move_sample = []
    
    # loop trough each emg data file
    for file in emg_files:
        data = np.load(file)
        keys = list(data.keys())
        freqs = data['freqs'] 
        
        if 'psd_rest' in keys:
            psd_rest = data['psd_rest']
            rest_time = data['rest_time']
            
        if 'psd_task' in keys:
            psd_task = data['psd_task']
            task_time = data['task_time']
        
        # add each psds for this subject in a list that we will average afterwards
        # add the len of the task for this file so we can create weight for averaging
        psd_rests.append(psd_rest) 
        rest_sample.append(len(rest_time))        
        if 'Hold' in file.__str__():
            psd_holds.append(psd_task)
            hold_sample.append(len(task_time))
            
        elif 'Move' in file.__str__():
            psd_moves.append(psd_task)
            move_sample.append(len(task_time))
    
    # if it is not a resting state file
    if 'Rest' not in file.__str__():  
        # weight = the actual rest sample divided by the sum of the of all condition
        weight_holds = np.array(hold_sample) / sum(hold_sample)
        weight_moves = np.array(move_sample) / sum(move_sample)
            
        # average all subjects files using the weights
        weighted_average_hold = np.average(psd_holds, axis=0, weights=weight_holds)
        weighted_average_move = np.average(psd_moves, axis=0, weights=weight_moves)
            
        # append the subjects files to the big average list 
        all_ave_psds_hold.append(weighted_average_hold)
        all_ave_psds_move.append(weighted_average_move)

    weight_rests = np.array(rest_sample) / sum(rest_sample)    
    weighted_average_rest = np.average(psd_rests, axis=0, weights=weight_rests)
    all_ave_psds_rest.append(weighted_average_rest)
        
# %%

# Calculate mean and standard deviation of the EMG signal during rest
mean_emg_rest = np.mean(all_ave_psds_rest)
std_emg_rest = np.std(all_ave_psds_rest)

# get the total number of subject for each condition 
total_rest = len(all_ave_psds_rest)
total_hold = len(all_ave_psds_hold)
total_move = len(all_ave_psds_move)

# for plotting each line of each psds --> repeat the frequency and reshape it for each psds
all_freqs_rest = np.tile(freqs, reps=len(subjects)).reshape(len(subjects), len(freqs))
all_freqs_hold = np.tile(freqs, reps=total_hold).reshape(total_hold, len(freqs)) 
all_freqs_move = np.tile(freqs, reps=total_move).reshape(total_move, len(freqs))

# get the mean for each condition
mean_rest = np.mean(np.vstack(all_ave_psds_rest), axis=0)
mean_hold = np.mean(np.vstack(all_ave_psds_hold), axis=0)
mean_move = np.mean(np.vstack(all_ave_psds_move), axis=0)

# get the std for each condition 
std_rest = np.std(np.vstack(all_ave_psds_rest), axis=0)
std_hold = np.std(np.vstack(all_ave_psds_hold), axis=0)
std_move = np.std(np.vstack(all_ave_psds_move), axis=0)

# std factor for each conditon (for plotting purposes fill between) very high variation in HOLD
std_factor_rest = 1 / total_rest
std_factor_hold = 1 / total_hold
std_factor_move = 1 / total_move 

# plot arguments 
plot_line = False # plot all lines for one condition
rest_color = 'green'
hold_color = 'darkorange'
move_color = 'blue'

plt.figure()
plt.plot(freqs, mean_rest, label='Rest', color=rest_color, alpha=0.7)
if plot_line:
    plt.plot(all_freqs_rest.T, np.array(all_ave_psds_rest).T, color=rest_color, alpha=0.2)
plt.fill_between(freqs, mean_rest - std_factor_rest * std_rest, mean_rest + std_factor_rest * std_rest, color=rest_color, alpha=0.2)

# plt.figure()
plt.plot(freqs, mean_hold, label='Hold', alpha=0.7, color=hold_color)
if plot_line:
    plt.plot(all_freqs_hold.T, np.array(all_ave_psds_hold).T, color=hold_color, alpha=0.2)
plt.fill_between(freqs, mean_hold - std_factor_hold * std_hold, mean_hold + std_factor_hold * std_hold, color=hold_color, alpha=0.2)

# plt.figure()
plt.plot(freqs, mean_move, label='Move', color=move_color, alpha=0.7)
if plot_line:
    plt.plot(all_freqs_move.T, np.array(all_ave_psds_move).T, color=move_color, alpha=0.2)    
plt.fill_between(freqs, mean_move - std_factor_move * std_move, mean_move + std_factor_move * std_move, color=move_color,alpha=0.2)

plt.legend()
plt.savefig('./figures/sub-GrandAverageEMG.jpg', dpi=300)
plt.show()