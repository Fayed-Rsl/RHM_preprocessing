# import the main libraries required for the preprocessing
import mne 
from mne_bids import read_raw_bids
from pathlib import Path
import pickle
import numpy as np 
# import homemade function 
from technical_validation_utils import subject_files
mne.set_log_level(verbose='CRITICAL') # reduce verbose output
# %%
# loop trough all the subject_files
for n_file, bids in enumerate(subject_files):        
    # create filename to save the var and indices for further group analysis and image for visual check
    var_basename = bids.copy().update(suffix='var', extension='.pkl', split=None, check=False).basename

    # subject path, if the directory does not already exist make one
    sub_path = Path(f'./var/sub-{bids.subject}/')
    sub_path.mkdir(parents=True, exist_ok=True)     
    
    # combine full fname for saving 
    var_path = Path(f'{sub_path}/{var_basename}')

    # # check if the files already exist, then the analyse has already been made then skip to the next file
    if var_path.exists():
        # print(f'{bids} already processed')
        continue
    
    # read the raw file
    raw = read_raw_bids(bids)

    # pick only the gradiometer channels
    raw.pick_types(meg='grad')
    raw.load_data()

    # calculate the sensors variance over time
    sensor_variances = np.var(raw._data, axis=1)
    
    # save the variance    
    save_var_dict = {'var':sensor_variances, 'meg_info':raw.info}
    with open(var_path, 'wb') as f: 
        pickle.dump(save_var_dict, f)
    print(f'Done with file n°{n_file} | {bids.basename}')

print('Calculating variance Done.')