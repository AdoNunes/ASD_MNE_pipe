#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:21:06 2019

@author: adonay

test MNEprepro class
"""

import appnope
appnope.nope()

from MNEprepro import MNEprepro
import time
import socket
import glob
import sys
import os.path as op

# %% Create variables for class object
paths_dic = {  # "root": "/Volumes/Data_projec/data/REPO/MEG_repo",
        "root": "~/Desktop/projects/MNE/data",
        "meg": "MEG",
        "subj_anat": 'anatomy',
        "out": "~/Desktop/projects/MNE/data_prep"
    }

Host = (socket.gethostname())

if Host == 'owners-MacBook-Pro.local':
    paths_dic['root'] = "/Volumes/4TB_drive/projects/MEG_repo/MEG_children_rs"
    paths_dic['out'] = "~/Desktop/projects/MNE/data_prep"
elif Host == 'sc-155028' or 'sc-155014':
    paths_dic['root'] = "~/Desktop/MEG_children_rs"
    paths_dic['out'] = "~/Desktop/projects/MNE/data_prep"

subject = '18011045C'
experiment = 'Flanker'

pth_tmp = op.join(op.expanduser(paths_dic["root"]), '18011*')
Subj_list = glob.glob(pth_tmp)

for iSubj in Subj_list:

    subject = op.basename(iSubj)
    print('Preprocessing subject: ' + subject)

    # %% Create Class object
    try:
        raw_prepro = MNEprepro(subject, experiment, paths_dic)
    except IndexError:
        continue

    # %% Detect and reject bad channels
    raw_prepro.detect_bad_channels(zscore_v=4, overwrite=False)

    # %% Detect and reject moving periods
    raw_prepro.detect_movement()
sys.exit()
# %% Muscle artifacts
raw_prepro.detect_muscle(overwrite=False, plot=True)

# %%Run
raw_prepro.run_ICA(overwrite=True)
# %%



# %% Events
#events, event_id  = raw_prepro.get_events(plot=1)


"""
#ica.exclude += eog_inds
#if ica_fname is None:
#    ica_fname = raw_f._filenames[0][:-4] + '-pyimpress-ica.fif'
#ica.save(ica_fname)
#return ica, ica_fname
#
#
#ecg_epochs = mne.preprocessing.create_ecg_epochs(raw_prepro.raw)
#ecg_epochs.plot_image(combine='mean')
#
#
#eog_evoked = mne.preprocessing.create_eog_epochs(raw_prepro.raw).average()
#eog_evoked.apply_baseline(baseline=(None, -0.2))
#eog_evoked.plot_joint()
#
#
#raw_copy.verbose = 'INFO'



ecchan = ['MLF13-4704',
 'MLF12-4704',
 'MLF14-4704',
 'MLF21-4704',
 'MLF22-4704',
 'MLF23-4704',
 'MLF24-4704',
 'MLF31-4704',
 'MLF25-4704',
 'MRT51-4704',
 'MLT51-4704',
 'MLT41-4704',
 'MLT31-4704',
 'MLT21-4704']
"""
