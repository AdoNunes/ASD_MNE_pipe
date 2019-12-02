#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:21:06 2019

@author: adonay

test MNEprepro class
"""

from MNEprepro import MNEprepro
import glob
import sys
import os.path as op

from m_set_directories import set_paths

paths_dic = set_paths()

subject = '18011045C'  # example
# options 'Movie', 'CarTask', 'Flanker'
experiment = 'CarTask'

# Get list of sibjects with experiment recording
pth_tmp = op.join(op.expanduser(paths_dic["root"]), '18011*', '*' +
                  experiment + '*')
Subj_list = [op.dirname(x) for x in sorted(glob.glob(pth_tmp))]


save_epo_no_musc = True

# %%


def piepline(iSubj):
    subject = op.basename(iSubj)
    print('Preprocessing subject: ' + subject)

    # %% Create Class object
    try:
        raw_prepro = MNEprepro(subject, experiment, paths_dic)
    except IndexError:
        return
#
#    # %% Detect and reject bad channels
#    raw_prepro.detect_bad_channels(zscore_v=4, overwrite=False)
#
#    # %% Detect and reject moving periods
#    raw_prepro.detect_movement(plot=True)
#
#    # %% Muscle artifacts
#    if save_epo_no_musc is not True:
#        raw_prepro.detect_muscle(overwrite=False, plot=True)
#
#    # %%Run ICA
#    try:
#        raw_prepro.run_ICA(overwrite=False)
#        raw_prepro.plot_ICA()
#    except RuntimeError:
#        return
#
#    # %%Create epochs
##    try:
##        raw_prepro.epoching(tmin=-0.7, tmax=0.7)
##    except ValueError:
##        return

    # %%Create forward mddelling
    raw_prepro.src_modelling(overwrite=False)

# %%


[piepline(iSubj) for iSubj in Subj_list[0:1]]


sys.exit()
############### TEMP ##############
import mne
from mne import make_forward_solution
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.connectivity import spectral_connectivity





FS_subj = op.join(paths_dic['FS'], subject)

spacing = 'oct5'
fname_trans = op.join(FS_subj, subject + '-trans.fif')
fname_bem = op.join(FS_subj, '%s-bem_sol.fif' % subject)
fname_src = op.join(FS_subj, 'bem', '%s-src.fif' % spacing)





fwd = make_forward_solution(raw_prepro.raw.info, fname_trans, src, fname_bem)
mne.write_forward_solution(fname_fwd, fwd)


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
