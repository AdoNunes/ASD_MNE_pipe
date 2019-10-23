#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 08:50:35 2019

@author: adonay

See effects of MXF head correction
"""


import appnope
appnope.nope()

from MNEprepro import MNEprepro
import time
import mne
import numpy as np
import socket
import os.path as op
import glob

start = time.time()
pause_for = 0.0005
time.sleep(pause_for)
time_past = time.time() - start
print(f'took {time_past:.4f} instead of {pause_for:.4f}')

if time_past > pause_for*3:
    import appnope
    appnope.nope()

# %%
    
if Host == 'owners-MacBook-Pro.local':
    paths_dic['root'] = "~/Desktop/projects/MNE/data"
    paths_dic['out'] = "~/Desktop/projects/MNE/data_prep"
elif Host == 'sc-155014':
    paths_dic['root'] = "~/Desktop/REPO/MEG_REPO/MEG_children"
    paths_dic['out'] = "~/Desktop/projects/MNE/data_prep"


# %% FIST WITH RAW OBJ ONLY
#######################################
subject = '18011045C'
raw_ds = op.join(op.expanduser(paths_dic['root']) , subject ,
                 subject + '_Flanker_20190923_02.ds')

raw = mne.io.read_raw_ctf(raw_ds)

events = mne.find_events(raw, stim_channel='UPPT001')

event_id = {'Incongruent/Left': 3, 'Incongruent/Right': 5,
            'Congruent/Left': 4, 'Congruent/Right': 6,
            'Reward/Face': 7, 'Reward/Coin': 8,
            'Reward/Neut_FaceTRL': 9, 'Reward/Neut_CoinTRL': 10}

picks = mne.pick_types(raw.info, meg=True, ref_meg=False)

epochs = mne.Epochs(raw, events, event_id, tmin=-0.1, tmax=.5,
                    baseline=(None, 0), picks=picks, preload=True)

# plot raw average
topomap_args = dict(vmin=-300, vmax=300)
evoked_con = epochs['Incongruent'].average()
evoked_con.plot_joint(title='No prepro', topomap_args=topomap_args)


# HM compensation
pos = mne.chpi._calculate_head_pos_ctf(raw)

raw_sss = raw.copy().apply_gradient_compensation(0)

destination = np.median(pos[..., 4:7], axis=0)

raw_sss = mne.preprocessing.maxwell_filter(raw_sss, destination=destination, head_pos=pos)

epochs_HM = mne.Epochs(raw_sss, events, event_id, tmin=-0.1, tmax=.5,
                       baseline=(None, 0), picks=picks, preload=True)
evoked_HM = epochs_HM['Incongruent'].average()
evoked_HM.plot_joint(title='Moving: movement compensated', topomap_args=topomap_args)

## %% Create variables for class object
#paths_dic = {  # "root": "/Volumes/Data_projec/data/REPO/MEG_repo",
#        "root": "~/Desktop/projects/MNE/data",
#        "meg": "MEG",
#        "subj_anat": 'anatomy',
#        "out": "~/Desktop/projects/MNE/data_prep"
#    }
#
#subject = '18011045C'
#experiment = 'Flanker'
#
## %% Create Class object
#raw_prepro = MNEprepro(subject, experiment, paths_dic)
#
## %% Detect and reject bad channels
#raw_prepro.detectBadChannels(save_csv=True)
#
## %% Detect and reject moving periods
##raw_prepro.detectMov()
#
#
## %% epoch data
#events = mne.find_events(raw_prepro.raw, stim_channel='UPPT001')
#
#print('Found %s events, first five:' % len(events))
#print(events[:5])
#
#event_id = {'Incongruent/Left': 3, 'Incongruent/Right': 5,
#            'Congruent/Left': 4, 'Congruent/Right': 6,
#            'Reward/Face': 7, 'Reward/Coin': 8,
#            'Reward/Neut_FaceTRL': 9, 'Reward/Neut_CoinTRL': 10}
#
#picks = mne.pick_types(raw_prepro.raw.info, meg=True, ref_meg=False)
#
#epochs = mne.Epochs(raw_prepro.raw, events, event_id, tmin=-0.1, tmax=.5,
#                    baseline=(None, 0), picks=picks, preload=True)
#
## plot average
#
#evoked_con = epochs['Congruent'].average()
#evoked_con.plot()
#evoked_con.plot_topomap(title='Stationary')
#
## %% correct head movement
#pos = mne.chpi._calculate_head_pos_ctf(raw_prepro.raw)
#
#raw_sss = mne.preprocessing.maxwell_filter(raw_prepro.raw, head_pos=pos)
#evoked_raw_mc = mne.Epochs(raw_sss, events, 1, -0.2, 0.8).average()
#evoked_raw_mc.plot_topomap(title='Moving: movement compensated', **topo_kwargs)
#
###############################
#
#mne.viz.plot_head_positions(pos, mode='traces', destination=None, 
#                            info=raw_prepro.raw.info);
#
## %% Create ICA components
#raw_prepro.run_ICA(overwrite=False)
#raw_prepro.plot_ICA()
#print("didnt stoppppp")
## %% Save ICA components
#raw_prepro.save_ICA(overwrite=False)

