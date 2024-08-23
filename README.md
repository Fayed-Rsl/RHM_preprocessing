
# RestHoldMove preprocessing

This repository has been made to help searcher to have a quick start on analysing the RestHoldMove dataset available here in openneuro:
[RestHoldMove dataset](https://openneuro.org/datasets/ds004907/versions/1.2.1).

## Introduction
This dataset contains data from externalized DBS patients undergoing simultaneous MEG - STN LFP recordings with (MedOn) and without (MedOn) dopaminergic medication. It has two movement conditions: 1) 5 min of rest followed by static forearm extension (hold) and 2) 5 min of rest followed by self-paced fist-clenching (move). The movement parts contain pauses. Some patients were recorded in resting-state only (rest). The project aimed to understand the neurophysiology of basal ganglia-cortex loops and its modulation by movement and medication.
For further information: https://www.nature.com/articles/s41597-024-03768-1

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install every library that are used in the repository.

```bash
cd RHM_preprocessing
pip install -r requirements. txt
```

## Usage

Define the root where you downloaded the dataset in the **config.py** file.
```python
bids_root = '/data/raw/hirsch/RestHoldMove_anon/' #redefine this 
``` 


#### Preprocessing

To perform basic preprocessing and connectivity analysis between MEG-STN in one subject block, run the **get_started.py** script. (_Detailed instructions are provided within the script_)
```bash
python get_started.py
```

#### Technical Validations

The utils folder contains all the scripts used to make the technical validations described in [_TBA_].
You can run these scripts to perform group analysis for UPDRS, EMG power, MEG-STN coupling.
Please note that you would need to run the **utils/run_<coherence/emg>.py** before starting the **utils/run_<coherence/emg>_average.py** scripts. 

Additionnaly the script **utils/technical_validation_utils.py** holds function that is used across the whole repository and has not been designed to be run individually.

## Figures
```bash
python run_updrs.py
```
![UPDRS](https://github.com/Fayed-Rsl/RHM_preprocessing/raw/master/utils/figures/sub-UPDRS.jpg)


```bash
python run_emg_average.py
```
![EMG](https://github.com/Fayed-Rsl/RHM_preprocessing/raw/master/utils/figures/sub-GrandAverageEMG.jpg)


```bash
python run_var_average.py
```
![EMG](https://github.com/Fayed-Rsl/RHM_preprocessing/raw/master/utils/figures/sub-GrandAverageVAR.jpg)


```bash
python run_coherence_average.py
```
![Coh Left](https://github.com/Fayed-Rsl/RHM_preprocessing/raw/master/utils/figures/sub-GrandAverageCOH-left.jpg)
![Coh Right](https://github.com/Fayed-Rsl/RHM_preprocessing/raw/master/utils/figures/sub-GrandAverageCOH-right.jpg)


```bash
source_reconstruction_example.m [matlab required]
```
![SOURCE](https://github.com/Fayed-Rsl/RHM_preprocessing/raw/master/utils/figures/sub-SourceReconstructionExample.jpg)

