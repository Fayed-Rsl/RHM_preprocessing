# This script has been made in order to quickly start analysing the MEG-LFP Dataset refering to this article: doi...
# For example purposes, preprocessing has been implemented only for a specific: subject, session, task, acquisition.
# One would need to adapt the following code in order to make the preprocessing on the whole dataset.
# Feel free to modify it.

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
%matplotlib qt 

# %%
bids_root = '/data/raw/hirsch/RestHoldMove_anon/'

# load UPDRS dataset 
updrs_on = Path(bids_root) / 'participants_updrs_on.tsv'
updrs_off = Path(bids_root) / 'participants_updrs_off.tsv'

on = pd.read_csv(updrs_on, sep='\t')
off = pd.read_csv(updrs_off, sep='\t')

on['status'] = 'ON'
off['status'] = 'OFF'


data = pd.concat([on[['participant_id', 'SUM', 'status']], off[['participant_id', 'SUM', 'status']]], ignore_index=True)
# remove the subject that only has missing val for statistical analysis, in order to have the same N of pop in both condition
data.dropna(how='any', inplace=True)

#%%
data_on = data[data['status'] == 'ON']['SUM']
data_off = data[data['status'] == 'OFF']['SUM']
statistic, p_value = ttest_ind(data_on, data_off)

# %%
fig, ax = plt.subplots(figsize=(10,6))

font = 20
sns.swarmplot(data=data, x='status', y='SUM', palette=['cornflowerblue', 'red'], alpha=0.8, ax=ax)

sns.violinplot(x='status', y='SUM', data=data, inner=None, palette=['cornflowerblue', 'red'], saturation=0.4, ax=ax)

annot = Annotator(ax, pairs=[('ON', 'OFF')], data=data, x='status', y='SUM')
annot.configure(test='t-test_ind', text_format='star', loc='inside')
annot.apply_and_annotate()

# plt.title('UPDRS scores and medication status')
plt.xlabel('')
plt.ylabel('')
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.tight_layout()
plt.yticks(np.arange(0, max(data['SUM']) + 10, 10))


# fig.savefig('figures/updrs_scores.jpg', dpi=300)
