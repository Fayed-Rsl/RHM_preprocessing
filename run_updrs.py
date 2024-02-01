# import the main libraries required for the preprocessing
import pandas as pd 
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

bids_root = '/data/raw/hirsch/RestHoldMove_anon/'

# UPDRS filename 
updrs_on = Path(bids_root) / 'participants_updrs_on.tsv'
updrs_off = Path(bids_root) / 'participants_updrs_off.tsv'

# load UPDRS dataset of all participants for each condition
on = pd.read_csv(updrs_on, sep='\t')
off = pd.read_csv(updrs_off, sep='\t')

# add a status to the dataframe
on['status'] = 'ON'
off['status'] = 'OFF'

# combine the ON and OFF dataset into one.
data = pd.concat([on[['participant_id', 'SUM', 'status']], off[['participant_id', 'SUM', 'status']]], ignore_index=True)

# remove the subject that only has missing value for statistical analysis, in order to have the same N of pop in both condition
data.dropna(how='any', inplace=True)

# make the sum of the UPDRS in both condition
data_on = data[data['status'] == 'ON']['SUM']
data_off = data[data['status'] == 'OFF']['SUM']

# create a figure 
fig, ax = plt.subplots(figsize=(10,6))

# font of text, ticks, label 
font = 20 

# plot each point for both condition
sns.swarmplot(data=data, x='status', y='SUM', palette=['cornflowerblue', 'red'], alpha=0.8, ax=ax)

# add a violin plot
sns.violinplot(x='status', y='SUM', data=data, inner=None, palette=['cornflowerblue', 'red'], saturation=0.4, ax=ax)


# compute t test 
statistic, p_value = ttest_ind(data_on, data_off)

# add stars if it is significant using statannotations library --> not necessary can comment this part if not wanted
# from statannotations.Annotator import Annotator 
# annot = Annotator(ax, pairs=[('ON', 'OFF')], data=data, x='status', y='SUM')
# annot.configure(test='t-test_ind', text_format='star', loc='outside')
# annot.apply_and_annotate()

# plt.title('UPDRS scores and medication status')
plt.xlabel('') # add the label later ... 
plt.ylabel('')
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.tight_layout()
plt.yticks(np.arange(0, max(data['SUM']) + 10, 10))

# save the figure in the current folder
fig.savefig('./figures/sub-UPDRS.jpg', dpi=300)