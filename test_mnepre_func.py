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

start = time.time()
pause_for = 0.0005
time.sleep(pause_for)
time_past = time.time() - start
print(f'took {time_past:.4f} instead of {pause_for:.4f}')

if time_past > pause_for*3:
    import appnope
    appnope.nope()


# %% Create variables for class object
paths_dic = {  # "root": "/Volumes/Data_projec/data/REPO/MEG_repo",
        "root": "~/Desktop/projects/MNE/data",
        "meg": "MEG",
        "subj_anat": 'anatomy',
        "out": "~/Desktop/projects/MNE/data_prep"
    }

Host = (socket.gethostname())

if Host == 'owners-MacBook-Pro.local':
    paths_dic['root'] = "~/Desktop/projects/MNE/data"
    paths_dic['out'] = "~/Desktop/projects/MNE/data_prep"
elif Host == 'sc-155028' or 'sc-155014':
    paths_dic['root'] = "~/Desktop/REPO/MEG_repo/MEG_children"
    paths_dic['out'] = "~/Desktop/projects/MNE/data_prep"

subject = '18011040A'
experiment = 'Movie'

# %% Create Class object
raw_prepro = MNEprepro(subject, experiment, paths_dic)
events, event_id  = raw_prepro.get_events(plot=1)

# %% Detect and reject bad channels
raw_prepro.detectBadChannels(save_csv=True)

# %% Detect and reject moving periods
#raw_prepro.detectMov()

# %% Create ICA components
raw_prepro.run_ICA(overwrite=False)
raw_prepro.plot_ICA()
print("didnt stoppppp")
# %% Save ICA components
raw_prepro.save_ICA(overwrite=False)



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
