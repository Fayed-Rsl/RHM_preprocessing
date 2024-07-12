%% LCMV Source Reconstruction example script.[Helped provided by Marius Krösche]
clearvars;
clc;

% path settings
ft_path = '/home/rasfay01/fieldtrip-20230503'; %fieltrip path
addpath(ft_path);
ft_defaults;
subject_path = '/data/raw/hirsch/RestHoldMove_anon/sub-0cGdk9/ses-PeriOp/'; % example for only one subject and one MEG file
meg_file = [subject_path, 'meg/sub-0cGdk9_ses-PeriOp_task-HoldL_acq-MedOff_run-1_meg.fif'];

% load variable: headmodel = hdm and sourcemodel = grid  
load([subject_path, 'headmodel/sub-0cGdk9_ses-PeriOp_headmodel.mat']); % var=hdm
load([subject_path, 'sourcemodel/sub-0cGdk9_ses-PeriOp_sourcemodel.mat']); % var = grid

% load standard templates for sourcemodel and mri 
load(fullfile([ft_path, '/template/sourcemodel/standard_sourcemodel3d4mm.mat']));
standard_sourcemodel = sourcemodel; clear sourcemodel; % rename the variable and clear the old one
standard_mri = ft_read_mri((fullfile([ft_path, '/template/anatomy/single_subj_T1.nii'])));


%% visualisation of the individual sourcemodel and the headmodel
figure;
ft_plot_headmodel(hdm,'edgecolor','none','facecolor','skin','facealpha',0.5);
hold on
view([90,0]);
ft_plot_mesh(grid.pos(grid.inside,:));

%% visualisation of the headmodel over the MEG sensors
sensors = ft_read_sens(meg_file,'senstype','meg');
sensors = ft_convert_units(sensors, 'cm');
figure
ft_plot_headmodel(hdm);
hold on
view([90,0]);
ft_plot_sens(sensors,'sensors','b*');

%% load the MEG data and apply basic preprocessing   
cfg = [];
cfg.continuous = 'yes';
cfg.dataset = meg_file;       
cfg.channel = 'megplanar';
data = ft_preprocessing(cfg);

% 1 Hz high pass filter
cfg = [];
cfg.hpfilter = 'yes'; 
cfg.hpfreq = 1;
cfg.hpfilttype = 'firws';
data = ft_preprocessing(cfg, data);

% Apply a notch filter to remove power line noise 
cfg = [];
cfg.bsfilter = 'yes';
cfg.bsfreq = [48 52; 98 102; 148 152; 198 202; 248 252];
data = ft_preprocessing(cfg, data);

% downsample from 2000 to 500 Hz
cfg = [];
cfg.resamplefs = 500;
data = ft_resampledata(cfg, data);
     
%% LCMV Beamforming
% cut the data into 1 sec trials
cfg = [];
cfg.length = 1;
CovData = ft_redefinetrial(cfg,data);

% change the time-field to calculate the covariance matrix at the same time for each epochs 
CovData.time(:) = CovData.time(1);

% the covariance matrix for the spatial filters
cfg = [];
cfg.covariance = 'yes';
avg = ft_timelockanalysis(cfg,CovData);

% compute spatial filter using a LCMV beamformer
cfg = [];
cfg.method = 'lcmv';
cfg.lcmv.lambda = '5%';
cfg.headmodel = hdm;
cfg.sourcemodel = grid;
cfg.lcmv.weightnorm = 'unitnoisegain';
cfg.lcmv.projectmom = 'yes';
cfg.lcmv.reducerank = 2; % typical value for MEG
cfg.lcmv.keepfilter = 'yes';
cfg.lcmv.projectnoise = 'yes';
source_time = ft_sourceanalysis(cfg,avg);

% Adapt size(filter) to match .pos, by filling empty values by zeros
source_time.avg.filter(cellfun('isempty',source_time.avg.filter)) = {zeros(1,length(source_time.avg.label))};
spatial_filt = cell2mat(source_time.avg.filter); %spatial filter for subject{i}, Sources X Channels

% Security check. zero-line sources are not part of a grid point.
if any( sum( abs( spatial_filt( source_time.inside(:) > 0 ,: ) ) ,2 ) == 0 )
    error(['At least one source that contributes to the calculation of parcel activation contain zero-line channel-weights,' newline 'which results in a zero-line time-course. However this source should not contribute to the parcel.'])
end

% Prepare to apply filter to the whole data. This might requires a lot of RAM, 200GB.
% Consider downsampling even further or select only part of the data that is of interest.
trial_source = [];
for fl = 1:length(data.trial)
    trial_source{fl} = spatial_filt * data.trial{fl}; 
end

%% plot the source on the standard_mri
% interpolate the source
cfg            = [];
cfg.parameter  = 'pow';
source_time_standard_pos = source_time; % create a copy of the source time
source_time_standard_pos.pos = standard_sourcemodel.pos; % change position to standard pos
source_inter = ft_sourceinterpolate(cfg, source_time_standard_pos, standard_mri);

% plot 3 orthogonal slices
cfg = [];
cfg.method        = 'ortho';
cfg.funparameter  = 'pow';
cfg.maskparameter = cfg.funparameter;
cfg.funcolormap    = 'jet';
cfg.location = [-22 -20 36]; 
ft_sourceplot(cfg, source_inter);