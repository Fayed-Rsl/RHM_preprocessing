# import the main libraries required for the preprocessing
import mne 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os 
import pickle
from scipy.stats import zscore
# import homemade function 
from technical_validation_utils import (parula_map, interp, subjects)
mne.set_log_level(verbose='CRITICAL') # reduce verbose output
# %%
# NB: run_var.py must has been runned before running the average
var_dir = [Path('./var/') / sub for sub in subjects]
assert all(file.exists() for file in var_dir), 'Please run_var.py. Path has not been created yet.'

# store variable for vars 
all_vars = []
got_info = False # check if we stored the MEG info with 204 sensors
highest = None

for n_sub, subject_dir in enumerate(var_dir):
    
    # variance files
    var_files = [subject_dir / file for file in os.listdir(subject_dir) if 'var' in file]
    var_files.sort() 

    # store all variance for one subjects to average over
    var_list = []

    # loop trough each var file
    for var_file in var_files:
        # read the file
        with open(var_file, 'rb') as f:
            var_struct = pickle.load(f)
            
        # get the data of the variance for each sensors --> shape should be (N_sensors) 
        var = var_struct['var']
        meg_info = var_struct['meg_info']
        orig_sensors = meg_info.ch_names.copy()
        sensors = meg_info.ch_names

        # calculate mean and standard deviation of variances
        mean_variance = np.mean(var)
        std_variance = np.std(var)
        
        # define the number of standard deviations threshold
        X = 2
        
        # filter out sensors with variances more than X standard deviations from the mean
        filtered_indices = np.abs(var - mean_variance) <= X * std_variance
        filtered_variances = var[filtered_indices]
        filtered_sensors = list(np.array(sensors)[filtered_indices])
        ind = np.arange(len(filtered_indices)) # we need integer not boolen when using mne.pick_info
        sel = ind[filtered_indices]
        
        meg_info = mne.pick_info(meg_info, sel=sel) # pick only selected sensors
        sensors = filtered_sensors.copy() # re assign the sensors to the filtered sensors

        # initialise highest
        if highest == None:
            highest = len(sensors) 
        # assign highest iteratively if not highest previously, in order to collect the maximum of meg_info
        elif highest < len(sensors):
            highest = len(sensors) 
        
        if len(sensors) == highest and got_info != True:
            meg_info_max = meg_info
            got_info = False
        
        var_list.append(filtered_variances)

    # interpolate the missing sensors and average on files axis and append it to the grand average list
    var_list = (interp(var_list)) 
    var_list = zscore(var_list) # normalise the variance of a subject so it doesn't lead the grand average
    all_vars.append(var_list)

# %%
# plot the grand average variance
grand_average_variance = (interp(all_vars))
# get the min and max for the vlim
vmin = np.min(grand_average_variance)
vmax = np.max(grand_average_variance)

fig, ax = plt.subplots(figsize=(8, 6)) 
im, _ = mne.viz.plot_topomap(grand_average_variance, meg_info_max, axes=ax,show=False, vlim=(vmin, vmax), cmap=parula_map, ch_type='grad')   

# add colorbar
ax_x_start = 0.8
ax_x_width = 0.015
ax_y_start = 0.3
ax_y_height = 0.4
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)

fig.savefig('./figures/sub-GrandAverageVAR.jpg', dpi=300)