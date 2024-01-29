import numpy as np
import matplotlib.pyplot as plt 
import mne
from matplotlib.colors import LinearSegmentedColormap
from mne_bids import BIDSPath, read_raw_bids
import pandas as pd
import mne_connectivity
from matplotlib.backends.backend_pdf import PdfPages
from scipy import interpolate
    
# get the onset and the duration for task (e.g., 'MoveL') event
# resting state period marked and the selected task.
# note that it is also possible to adapt the script to extend the rest period to everything that it is not marked as the task.
# but here we only include the period that were marked as rest, but not the rest period between the task aswell.
def get_raw_condition(raw, conds):
    '''    
    Parameters
    ----------
    raw : mne.io.Raw
        raw object
    conds : list
        list of conditions
    Returns
    -------
    tuple of rest and task segments
    if no event found in the raw.annotations --> the value will be None.
    '''    
    
    conditions = conds.copy() # create a copy of the list so we can modify it without changing the original list 
    allowed_conditions = [['rest', 'HoldL'], ['rest', 'MoveL'], ['rest', 'HoldR'], ['rest', 'MoveR'], ['rest']]
    assert conditions in allowed_conditions, f'Conditions should be in {allowed_conditions}'
    
    # initialise the segments by None
    task_segments = None 
    rest_segments = None
    
    # check that the raw actually have resting state not only task
    if 'rest' not in raw.annotations.description:
        conditions.remove('rest')

    for task in conditions:
        # get the onset and the duration of the event 
        segments_onset = raw.annotations.onset[raw.annotations.description == task]
        segments_duration = raw.annotations.duration[raw.annotations.description == task]
        
        # substract the first_sample delay in the onset
        segments_onset = segments_onset - (raw.first_samp / raw.info['sfreq'])
        
        # loop trough the onset and duration to get only part of the raw that are in the task
        for i, (onset, duration) in enumerate(zip(segments_onset, segments_duration)):
            # if it is the first onset, initialise the raw object storing all of the task segments
            if task != 'rest':
                if i == 0:
                    task_segments = raw.copy().crop(tmin=onset, tmax=onset+duration)
                # otherwise, append the segments to the existing raw object
                else:
                    task_segments.append([raw.copy().crop(tmin=onset, tmax=onset+duration)])
            # do the same for rest
            else:
                if i == 0:
                    rest_segments = raw.copy().crop(tmin=onset, tmax=onset+duration)
                else:
                    rest_segments.append([raw.copy().crop(tmin=onset, tmax=onset+duration)])
    return rest_segments, task_segments


# function to find the selected_emg during the task 
def get_selected_emg(emgs, ch_names, side):
    '''
    Parameters
    ----------
    emgs : np.array
        (4, n_times). #4 channels, 2 left - 2 right
    ch_names : list
        name of the EMG channels
    side : str
        left, right or find. to find the side of the task

    Returns
    -------
    selected_emg_mean : np.array
        the mean of the emgs that were used during the task
    selected_emg_label : str
        the side of the emgs
    ch_names : list
        the names of the emgs
    '''
    
    # slice the emgs with the sides.
    emg_left = emgs[2:]
    emg_right = emgs[:2]
    
    # find the side automatically using RMS     
    if side == 'find':
        rms_right = np.sqrt(np.mean(emg_right**2))
        rms_left = np.sqrt(np.mean(emg_left**2))
        selected_emg =  emg_left if rms_left > rms_right else  emg_right
        selected_emg_mean = np.mean(selected_emg, axis=0) # average of the 2 emgs channels
    
    # force the side of the emgs
    elif side == 'left':
        selected_emg_mean =  np.mean(emg_left, axis=0)
    elif side == 'right':
        selected_emg_mean =  np.mean(emg_right, axis=0)
    elif side == 'all':
        selected_emg_mean =  np.mean(emgs, axis=0)
    
    # make the correct label and channel names
    if side == 'find':
        selected_emg_label = 'left' if rms_left > rms_right else 'right' 
    else:
        selected_emg_label = side 
    if selected_emg_label == 'right':
        ch_names = ch_names[:2]
    elif selected_emg_label == 'left':
        ch_names = ch_names[2:]
        
    print(selected_emg_label, ch_names)
    return selected_emg_mean, selected_emg_label, ch_names

# function to get the emg power in the selected EMG side
def get_emg_power(raw, ch_names, side='find', fmin=2, fmax=45):
    emgs = raw.get_data()
    sfreq = raw.info['sfreq'] 
    selected_emg, selected_emg_label, ch_names = get_selected_emg(emgs, ch_names, side)
    psd, freqs = mne.time_frequency.psd_array_welch(selected_emg, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=2048)
    return psd, freqs, selected_emg_label, ch_names

# function to plot the power of emgs
def plot_power(psd, freqs, side, condition, color, ax):
    ax.plot(freqs, psd, label=f'{condition}', alpha=0.5, color=color)
    ax.legend()

# function for coherence, in order to preprocess the raw resting state and transform it to an epochs. 
def preprocess_raw_to_epochs(bids, downsample=200):
    '''
    Parameters
    ----------
    bids : mne_bids.BIDSPath
        BIDSPath object.
    downsample : int, optional
        resample the sampling frequency from 2000 to 200. The default is 200.
    Returns
    -------
    epochs : mne.io.Epochs
        epochs object
    '''
    
    # read the raw bids
    raw = read_raw_bids(bids)
    
    # we are only preprocessing resting state here, so if there is no rest in the annotations, just stop the task ...
    if 'rest' not in raw.annotations.description:
        print('No rest for', bids)
        return 

    # create a dict of all wanted channel types --> meg gradiometers and eeg
    channel_to_pick = {'meg':'grad', 'eeg': True} 
    
    # pick all channels types and exclude the bads one
    raw.pick_types(**channel_to_pick, exclude='bads')
    
    # create a bids path for the montage, check=False because this is not a standard bids file
    montage_path = BIDSPath(root=bids.root, session=bids.session, subject=bids.subject,
                            datatype='montage', extension='.tsv', check=False)
    
    # get the montage tsv bids file (first item on the list match)
    montage = montage_path.match()[0]
    
    # read the tsv file using pandas
    df = pd.read_csv(montage, sep='\t')
    
    # create a dictionary mapping old names to new names for right and left channels
    montage_mapping = {row['right_contacts_old']: row['right_contacts_new'] for idx, row in df.iterrows()} # right
    montage_mapping.update({row['left_contacts_old']: row['left_contacts_new'] for idx, row in df.iterrows()}) # left
    
    # remove in the montage mapping the channels that are not in the raw anymore (because they were bads)
    montage_mapping  = {key: value for key, value in montage_mapping.items() if key in raw.ch_names }
    
    # rename the channels using the montage mapping scheme EEG001 --> LFP-<side>-0
    raw.rename_channels(montage_mapping)

    # get list of lfp names and their side 
    lfp_name = [name for name in raw.ch_names if 'LFP' in name]
    lfp_right = [name for name in lfp_name if 'right' in name]
    lfp_left = [name for name in lfp_name if 'left' in name]
    
    # get LFP-side-contact then add the next contact number to the name to create the new bipolar name
    # e.g. LFP-right-0 and LFP-right-1 will become LFP-right-01
    if len(lfp_right) > 1: # we need at least 2 contact to create a bipolar scheme
        bipolar_right = [f'{lfp_right[i]}{lfp_right[i+1][-1]}' for i in range(len(lfp_right)-1)]
        ref_right = True
    else:
        ref_right = False # else do no re-reference this side
    
    if len(lfp_left) > 1:
        bipolar_left = [f'{lfp_left[i]}{lfp_left[i+1][-1]}' for i in range(len(lfp_left)-1)]
        ref_left = True
    else: 
        ref_left = False
        
    # from now on we need to load the data to apply filter and re-referencing
    raw.load_data()
    
    # set bipolar reference scheme for each respective side 
    if ref_right:
        raw = mne.set_bipolar_reference(raw, anode=lfp_right[:-1], cathode=lfp_right[1:], ch_name=bipolar_right)
        
    if ref_left:
        raw = mne.set_bipolar_reference(raw, anode=lfp_left[:-1], cathode=lfp_left[1:], ch_name=bipolar_left)

    # apply a 1Hz high pass firwin filter to remove slow drifts
    raw.filter(l_freq=1, h_freq=None, method='fir', n_jobs=-1)
    
    # after filtering, now do downsample to 200 from 2000. NB: downsample after filter so no aliasing introduced
    raw.resample(downsample)
    
    # crop the data and get the rest_segments only if the task is not already Rest otherwise, keep the raw data as it is
    if bids.task != 'Rest':
        raw, _ = get_raw_condition(raw, ['rest'])

    # create epochs of fixed length 2 seconds and 50% overlap.
    epochs = mne.make_fixed_length_epochs(raw, duration=2, overlap=1, preload=True)
    return epochs 


# make individual topomap 1_lfp*megs
def individual_lfpmeg_coh(seeds, targets, data, sfreq, fmin, fmax):
    individual_coh_list = [] # store the individual coherence in a list
    # loop over the seeds i.e., list of LFP sensors and do connectivity for one (seed) LFP against all targets (MEG)
    for seed in seeds:
        indices = mne_connectivity.seed_target_indices(seed, targets) 
        # compute spectral connectivity coherence with multitaper and append it to a list 
        coh_lfpmeg = mne_connectivity.spectral_connectivity_epochs(data, fmin=fmin, fmax=fmax,
                   method='coh', mode='multitaper', sfreq=sfreq, indices=indices, n_jobs=-1)
        individual_coh_list.append(coh_lfpmeg)

    return individual_coh_list

def plot_coh_topo(coh, n_lfp, n_meg, meg_info, freqs_beta, ax, lfp_ind=[],average_type='all', dB=False):
    # get the data of the connectivity coherence --> shape should be (N_connection * N_freqs)
    coh_data = coh.get_data() 
    coh_freqs = np.array(coh.freqs)

    # reshape the data according to (N_lfp * N_meg * N_freqs)
    coh_data = coh_data.reshape(n_lfp, n_meg, len(coh_freqs))    

    # average the connectivity on LFP axis channel, average type: either all LFP, only left or right side
    if average_type == 'all':        
        coh_average = np.mean(coh_data, axis=0)
    elif average_type == 'side':
        # assert that the lfp_ind is not empty
        assert len(lfp_ind) > 0, 'Please specifiy the index of the lfp side.'
        coh_average = np.mean(coh_data[lfp_ind, :, :], axis=0) # get only the LFP on the requested side index
   
    # create a mne Spectrum array using MEG sensors information
    coh_spec = mne.time_frequency.SpectrumArray(coh_average, meg_info, freqs=coh_freqs)
    
    # fieldtrip colors <3, better than mne default colors! 
    coh_spec.plot_topomap(bands=freqs_beta, res=300, cmap=(parula_map, True), dB=dB, axes=ax)
    return ax

# plot indviidual topomap
def plot_individual_coh_topo(coh_list, lfp_ch_names, meg_info, freqs_beta, dB=False, show=True): 
    # calculate the number of LFP channels on the left and right
    left_channels = sum('left' in name for name in lfp_ch_names)
    right_channels = sum('right' in name for name in lfp_ch_names)
    
    # determine the number of columns dynamically
    num_columns = max(left_channels, right_channels)
    
    # create a figure of 2 rows: row1=LEFT, row2=RIGHTH and n_columns depending on how much LFP in that side
    fig, ax = plt.subplots(2, num_columns, figsize=(8, 6))
    row1, row2 = 0, 0 # initialise both rows to 0
    for idx, (coh, name) in enumerate(zip (coh_list, lfp_ch_names)):
        # get the data of the connectivity coherence --> shape should be (N_connection * N_freqs)
        coh_data = coh.get_data()
        
        # get the frequency
        coh_freqs = np.array(coh.freqs)
        
        # create a mne Spectrum array using MEG sensors information
        coh_spec = mne.time_frequency.SpectrumArray(coh_data, meg_info, freqs=coh_freqs)
    
        # plot for specific sides with fieldtriep color map <3 
        if 'left' in name:
            coh_spec.plot_topomap(bands=freqs_beta, res=300, cmap=(parula_map, True), dB=dB, axes=ax[0][row1])
            ax[0][row1].set_title(name)
            row1 += 1
    
        elif 'right' in name:
            coh_spec.plot_topomap(bands=freqs_beta, res=300, cmap=(parula_map, True), dB=dB, axes=ax[1][row2])
            ax[1][row2].set_title(name)
            row2 += 1
        
    # remove extra empty axes for each side
    for i in range(left_channels, num_columns):
        fig.delaxes(ax[0][i])
        
    for i in range(right_channels, num_columns):
        fig.delaxes(ax[1][i])
    
    if show:
        plt.show()


# interpolate function to have a list of interpolated coherence matrices with the same size
def interp(grand_ave):
    # make the grand ave list into an array
    array = np.array(grand_ave, dtype=object)
    # find the maximum number of sensors across all subjects
    max_sensors = max(matrix.shape[0] for matrix in array)
    
    # initialize a list to store interpolated coherence matrices
    interpolated_coherence = []
    
    # interpolate coherence matrices to have the same number of sensors
    for c in array:
        n_sensors = c.shape[0]
        
        # interpolate along the sensor axis
        x = np.arange(n_sensors)
        f = interpolate.interp1d(x, c, axis=0, kind='linear', fill_value='extrapolate')
        
        # create new axis for interpolated sensors
        new_x = np.linspace(0, n_sensors - 1, max_sensors)
        
        # interpolate coherence matrix
        interpolated = f(new_x)
        interpolated_coherence.append(interpolated)
        
    # average over all subjects axis
    inter_grand_ave = np.mean(interpolated_coherence, axis=0) 
    return inter_grand_ave


# function to save multiple figures into a PDF. basically to save the topos.
def save_multipage(filename, figs=None, dpi=300):
    '''saving multiple figure in one pdf'''
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf', dpi=dpi)
    pp.close()
    
    
# Matlab default color map          
cm_data = [[0.2422, 0.1504, 0.6603],
[0.2444, 0.1534, 0.6728],
[0.2464, 0.1569, 0.6847],
[0.2484, 0.1607, 0.6961],
[0.2503, 0.1648, 0.7071],
[0.2522, 0.1689, 0.7179],
[0.254, 0.1732, 0.7286],
[0.2558, 0.1773, 0.7393],
[0.2576, 0.1814, 0.7501],
[0.2594, 0.1854, 0.761],
[0.2611, 0.1893, 0.7719],
[0.2628, 0.1932, 0.7828],
[0.2645, 0.1972, 0.7937],
[0.2661, 0.2011, 0.8043],
[0.2676, 0.2052, 0.8148],
[0.2691, 0.2094, 0.8249],
[0.2704, 0.2138, 0.8346],
[0.2717, 0.2184, 0.8439],
[0.2729, 0.2231, 0.8528],
[0.274, 0.228, 0.8612],
[0.2749, 0.233, 0.8692],
[0.2758, 0.2382, 0.8767],
[0.2766, 0.2435, 0.884],
[0.2774, 0.2489, 0.8908],
[0.2781, 0.2543, 0.8973],
[0.2788, 0.2598, 0.9035],
[0.2794, 0.2653, 0.9094],
[0.2798, 0.2708, 0.915],
[0.2802, 0.2764, 0.9204],
[0.2806, 0.2819, 0.9255],
[0.2809, 0.2875, 0.9305],
[0.2811, 0.293, 0.9352],
[0.2813, 0.2985, 0.9397],
[0.2814, 0.304, 0.9441],
[0.2814, 0.3095, 0.9483],
[0.2813, 0.315, 0.9524],
[0.2811, 0.3204, 0.9563],
[0.2809, 0.3259, 0.96],
[0.2807, 0.3313, 0.9636],
[0.2803, 0.3367, 0.967],
[0.2798, 0.3421, 0.9702],
[0.2791, 0.3475, 0.9733],
[0.2784, 0.3529, 0.9763],
[0.2776, 0.3583, 0.9791],
[0.2766, 0.3638, 0.9817],
[0.2754, 0.3693, 0.984],
[0.2741, 0.3748, 0.9862],
[0.2726, 0.3804, 0.9881],
[0.271, 0.386, 0.9898],
[0.2691, 0.3916, 0.9912],
[0.267, 0.3973, 0.9924],
[0.2647, 0.403, 0.9935],
[0.2621, 0.4088, 0.9946],
[0.2591, 0.4145, 0.9955],
[0.2556, 0.4203, 0.9965],
[0.2517, 0.4261, 0.9974],
[0.2473, 0.4319, 0.9983],
[0.2424, 0.4378, 0.9991],
[0.2369, 0.4437, 0.9996],
[0.2311, 0.4497, 0.9995],
[0.225, 0.4559, 0.9985],
[0.2189, 0.462, 0.9968],
[0.2128, 0.4682, 0.9948],
[0.2066, 0.4743, 0.9926],
[0.2006, 0.4803, 0.9906],
[0.195, 0.4861, 0.9887],
[0.1903, 0.4919, 0.9867],
[0.1869, 0.4975, 0.9844],
[0.1847, 0.503, 0.9819],
[0.1831, 0.5084, 0.9793],
[0.1818, 0.5138, 0.9766],
[0.1806, 0.5191, 0.9738],
[0.1795, 0.5244, 0.9709],
[0.1785, 0.5296, 0.9677],
[0.1778, 0.5349, 0.9641],
[0.1773, 0.5401, 0.9602],
[0.1768, 0.5452, 0.956],
[0.1764, 0.5504, 0.9516],
[0.1755, 0.5554, 0.9473],
[0.174, 0.5605, 0.9432],
[0.1716, 0.5655, 0.9393],
[0.1686, 0.5705, 0.9357],
[0.1649, 0.5755, 0.9323],
[0.161, 0.5805, 0.9289],
[0.1573, 0.5854, 0.9254],
[0.154, 0.5902, 0.9218],
[0.1513, 0.595, 0.9182],
[0.1492, 0.5997, 0.9147],
[0.1475, 0.6043, 0.9113],
[0.1461, 0.6089, 0.908],
[0.1446, 0.6135, 0.905],
[0.1429, 0.618, 0.9022],
[0.1408, 0.6226, 0.8998],
[0.1383, 0.6272, 0.8975],
[0.1354, 0.6317, 0.8953],
[0.1321, 0.6363, 0.8932],
[0.1288, 0.6408, 0.891],
[0.1253, 0.6453, 0.8887],
[0.1219, 0.6497, 0.8862],
[0.1185, 0.6541, 0.8834],
[0.1152, 0.6584, 0.8804],
[0.1119, 0.6627, 0.877],
[0.1085, 0.6669, 0.8734],
[0.1048, 0.671, 0.8695],
[0.1009, 0.675, 0.8653],
[0.0964, 0.6789, 0.8609],
[0.0914, 0.6828, 0.8562],
[0.0855, 0.6865, 0.8513],
[0.0789, 0.6902, 0.8462],
[0.0713, 0.6938, 0.8409],
[0.0628, 0.6972, 0.8355],
[0.0535, 0.7006, 0.8299],
[0.0433, 0.7039, 0.8242],
[0.0328, 0.7071, 0.8183],
[0.0234, 0.7103, 0.8124],
[0.0155, 0.7133, 0.8064],
[0.0091, 0.7163, 0.8003],
[0.0046, 0.7192, 0.7941],
[0.0019, 0.722, 0.7878],
[0.0009, 0.7248, 0.7815],
[0.0018, 0.7275, 0.7752],
[0.0046, 0.7301, 0.7688],
[0.0094, 0.7327, 0.7623],
[0.0162, 0.7352, 0.7558],
[0.0253, 0.7376, 0.7492],
[0.0369, 0.74, 0.7426],
[0.0504, 0.7423, 0.7359],
[0.0638, 0.7446, 0.7292],
[0.077, 0.7468, 0.7224],
[0.0899, 0.7489, 0.7156],
[0.1023, 0.751, 0.7088],
[0.1141, 0.7531, 0.7019],
[0.1252, 0.7552, 0.695],
[0.1354, 0.7572, 0.6881],
[0.1448, 0.7593, 0.6812],
[0.1532, 0.7614, 0.6741],
[0.1609, 0.7635, 0.6671],
[0.1678, 0.7656, 0.6599],
[0.1741, 0.7678, 0.6527],
[0.1799, 0.7699, 0.6454],
[0.1853, 0.7721, 0.6379],
[0.1905, 0.7743, 0.6303],
[0.1954, 0.7765, 0.6225],
[0.2003, 0.7787, 0.6146],
[0.2061, 0.7808, 0.6065],
[0.2118, 0.7828, 0.5983],
[0.2178, 0.7849, 0.5899],
[0.2244, 0.7869, 0.5813],
[0.2318, 0.7887, 0.5725],
[0.2401, 0.7905, 0.5636],
[0.2491, 0.7922, 0.5546],
[0.2589, 0.7937, 0.5454],
[0.2695, 0.7951, 0.536],
[0.2809, 0.7964, 0.5266],
[0.2929, 0.7975, 0.517],
[0.3052, 0.7985, 0.5074],
[0.3176, 0.7994, 0.4975],
[0.3301, 0.8002, 0.4876],
[0.3424, 0.8009, 0.4774],
[0.3548, 0.8016, 0.4669],
[0.3671, 0.8021, 0.4563],
[0.3795, 0.8026, 0.4454],
[0.3921, 0.8029, 0.4344],
[0.405, 0.8031, 0.4233],
[0.4184, 0.803, 0.4122],
[0.4322, 0.8028, 0.4013],
[0.4463, 0.8024, 0.3904],
[0.4608, 0.8018, 0.3797],
[0.4753, 0.8011, 0.3691],
[0.4899, 0.8002, 0.3586],
[0.5044, 0.7993, 0.348],
[0.5187, 0.7982, 0.3374],
[0.5329, 0.797, 0.3267],
[0.547, 0.7957, 0.3159],
[0.5609, 0.7943, 0.305],
[0.5748, 0.7929, 0.2941],
[0.5886, 0.7913, 0.2833],
[0.6024, 0.7896, 0.2726],
[0.6161, 0.7878, 0.2622],
[0.6297, 0.7859, 0.2521],
[0.6433, 0.7839, 0.2423],
[0.6567, 0.7818, 0.2329],
[0.6701, 0.7796, 0.2239],
[0.6833, 0.7773, 0.2155],
[0.6963, 0.775, 0.2075],
[0.7091, 0.7727, 0.1998],
[0.7218, 0.7703, 0.1924],
[0.7344, 0.7679, 0.1852],
[0.7468, 0.7654, 0.1782],
[0.759, 0.7629, 0.1717],
[0.771, 0.7604, 0.1658],
[0.7829, 0.7579, 0.1608],
[0.7945, 0.7554, 0.157],
[0.806, 0.7529, 0.1546],
[0.8172, 0.7505, 0.1535],
[0.8281, 0.7481, 0.1536],
[0.8389, 0.7457, 0.1546],
[0.8495, 0.7435, 0.1564],
[0.86, 0.7413, 0.1587],
[0.8703, 0.7392, 0.1615],
[0.8804, 0.7372, 0.165],
[0.8903, 0.7353, 0.1695],
[0.9, 0.7336, 0.1749],
[0.9093, 0.7321, 0.1815],
[0.9184, 0.7308, 0.189],
[0.9272, 0.7298, 0.1973],
[0.9357, 0.729, 0.2061],
[0.944, 0.7285, 0.2151],
[0.9523, 0.7284, 0.2237],
[0.9606, 0.7285, 0.2312],
[0.9689, 0.7292, 0.2373],
[0.977, 0.7304, 0.2418],
[0.9842, 0.733, 0.2446],
[0.99, 0.7365, 0.2429],
[0.9946, 0.7407, 0.2394],
[0.9966, 0.7458, 0.2351],
[0.9971, 0.7513, 0.2309],
[0.9972, 0.7569, 0.2267],
[0.9971, 0.7626, 0.2224],
[0.9969, 0.7683, 0.2181],
[0.9966, 0.774, 0.2138],
[0.9962, 0.7798, 0.2095],
[0.9957, 0.7856, 0.2053],
[0.9949, 0.7915, 0.2012],
[0.9938, 0.7974, 0.1974],
[0.9923, 0.8034, 0.1939],
[0.9906, 0.8095, 0.1906],
[0.9885, 0.8156, 0.1875],
[0.9861, 0.8218, 0.1846],
[0.9835, 0.828, 0.1817],
[0.9807, 0.8342, 0.1787],
[0.9778, 0.8404, 0.1757],
[0.9748, 0.8467, 0.1726],
[0.972, 0.8529, 0.1695],
[0.9694, 0.8591, 0.1665],
[0.9671, 0.8654, 0.1636],
[0.9651, 0.8716, 0.1608],
[0.9634, 0.8778, 0.1582],
[0.9619, 0.884, 0.1557],
[0.9608, 0.8902, 0.1532],
[0.9601, 0.8963, 0.1507],
[0.9596, 0.9023, 0.148],
[0.9595, 0.9084, 0.145],
[0.9597, 0.9143, 0.1418],
[0.9601, 0.9203, 0.1382],
[0.9608, 0.9262, 0.1344],
[0.9618, 0.932, 0.1304],
[0.9629, 0.9379, 0.1261],
[0.9642, 0.9437, 0.1216],
[0.9657, 0.9494, 0.1168],
[0.9674, 0.9552, 0.1116],
[0.9692, 0.9609, 0.1061],
[0.9711, 0.9667, 0.1001],
[0.973, 0.9724, 0.0938],
[0.9749, 0.9782, 0.0872],
[0.9769, 0.9839, 0.0805]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

