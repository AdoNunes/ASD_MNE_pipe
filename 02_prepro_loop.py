#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:21:06 2019

@author: adonay

test MNEprepro class
"""

from MNEprepro import MNEprepro
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
    import appnope
    appnope.nope()
elif Host == 'sc-155028' or Host == 'sc-155014':
    paths_dic['root'] = "~/Desktop/REPO/MEG_repo/MEG_children_rs"
    paths_dic['out'] = "~/Desktop/projects/MNE/data_prep"
    import appnope
    appnope.nope()
elif Host == 'megryan.nmr.mgh.harvard.edu':
    path_gen = '/local_mount/space/megryan/2/users/adonay/projects/ASD/'
    paths_dic['root'] = path_gen +"/MEG_children_rs"
    paths_dic['out'] = path_gen + "/data_prep"

subject = '18011045C'

# options 'Movie', 'CarTask', 'Flanker'
experiment = 'CarTask'

pth_tmp = op.join(op.expanduser(paths_dic["root"]), '18011*')
Subj_list = sorted(glob.glob(pth_tmp))

# %%
for iSubj in Subj_list[3:]:

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
    raw_prepro.detect_movement(plot=False)

    # %% Muscle artifacts
    raw_prepro.detect_muscle(overwrite=False, plot=True)
    # %%Run
    raw_prepro.run_ICA(overwrite=False)
# %%
sys.exit()

raw_prepro.plot_ICA()
event_id, events = raw_prepro.get_events()
epochs = raw_prepro.epoching(event_id, events, tmin=-0.7, tmax=0.7)
epochs.save(paths_dic['out'] + '/epoched/' + subject + '-epo.fif')
sys.exit()
############### TEMP ##############
import mne
from mne import setup_volume_source_space, setup_source_space
from mne import make_forward_solution
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.connectivity import spectral_connectivity


paths_dic['FS'] = path_gen + "/Freesurfer_children"
FS_subj = op.join(paths_dic['FS'], subject)

spacing = 'oct5'
fname_trans = op.join(FS_subj, subject + '-trans.fif')
fname_bem = op.join(FS_subj, '%s-bem_sol.fif' % subject)
fname_src = op.join(FS_subj, 'bem', '%s-src.fif' % spacing)

src = mne.read_source_spaces(fname_src)
fwd = make_forward_solution(raw_prepro.raw.info, fname_trans, src, fname_bem)

cov = mne.compute_covariance(epochs, method='auto')

inv = mne.minimum_norm.make_inverse_operator(raw_prepro.raw.info, fwd, cov, loose=0.2)

evoked = epochs.average()
stc = mne.minimum_norm.apply_inverse(evoked, inv, lambda2=1. / 9.)

surfer_kwargs = dict(hemi='split', subjects_dir=paths_dic['FS'],
                     subject=subject, views=['lateral', 'medial'],
                     clim=dict(kind='value', lims=[80, 97.5,100]),time_viewer=True,
                     size=(2000, 2000), background='white', initial_time=0.1)

stc.plot(surface='inflated', **surfer_kwargs)






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

"""