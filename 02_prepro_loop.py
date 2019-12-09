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

# %%


def piepline(iSubj, opt):
    ''' Function for running preprocessing steps '''

    subject = op.basename(iSubj)
    print('Preprocessing subject: ' + subject)

    # %% Create Class object
    raw_prepro = MNEprepro(subject, experiment, paths_dic)

    # %% Detect and reject bad channels
    if opt['bad_chans'] is True:
        raw_prepro.detect_bad_channels(zscore_v=4, overwrite=False)
        raw_prepro.raw.load_data().interpolate_bads(origin=[0, 0, 0])
    # %% Detect and reject moving periods
    if opt['movement']is True:
        raw_prepro.detect_movement(overwrite=False, plot=False)

    # %% Detect and reject periods with muscle
    if opt['muscle'] is True:
        raw_prepro.detect_muscle(overwrite=False, plot=True)

    # %%Run ICA
    try:
        if opt['ICA_run'] is True:
            raw_prepro.run_ICA(overwrite=False)
        if opt['ICA_plot'] is True:
            raw_prepro.plot_ICA()
    except RuntimeError:
        return

    # %%Create epochs
    if opt['epoching'] is True:
        raw_prepro.epoching(overwrite=False, tmin=-0.7, tmax=0.7, plot=True)

    # %%Create forward mddelling
    if opt['src_model'] is True:
        raw_prepro.src_modelling(overwrite=False)
    return raw_prepro
# %%


options_run = dict()
options_run['bad_chans'] = True
options_run['movement'] = True
options_run['muscle'] = False
options_run['ICA_run'] = False
options_run['ICA_plot'] = True
options_run['epoching'] = True
options_run['src_model'] = False

import time
start = time.time()
raw_prepro = [piepline(iSubj, options_run) for iSubj in Subj_list[27:]]
end = time.time() - start

sys.exit()
############### TEMP ##############
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.connectivity import spectral_connectivity


outdir = paths_dic['data2src'] + '/*_' + experiment + '-epo.fif'
subj_epo = sorted(glob.glob(outdir))

crown = ['18011007', '18011008', '18011029', '18011032', '18011039']
subj_epo = [s for s in subj_epo if not any(ignore in s for ignore in crown)]

subj_ix = {}
subj_ix['all'] = list(range(len(subj_epo)))
subj_ix['ASD'] = [s for s in range(len(subj_epo)) if 'A' in op.basename(subj_epo[s])[8]]
subj_ix['CTR'] = [s for s in range(len(subj_epo)) if 'C' in op.basename(subj_epo[s])[8]]


# Get conditions names and create containers
evoked_all = dict()
conditions = mne.read_epochs(Subj_list_epo[0]).event_id.keys()
for c in conditions:
    evoked_all[c] = []

# Get evoked data
for iSubj in Subj_list_epo:
    epochs = mne.read_epochs(iSubj)
    for c in conditions:
        evoked_all[c].append(epochs[c].copy().average())

grp = ['all', 'ASD', 'CTR']
for g in grp:
    gix = subj_ix[g]

    for c in conditions:
        evoked_grp = [evoked_all[c][i] for i in gix]
        mne.combine_evoked(evoked_grp, 'nave').plot_joint(title=g + ' ' + c)



mne.viz.plot_arrowmap(evoked, evoked.info)














noise_cov = mne.compute_covariance(raw_prepro.epochs,tmax=0, method='auto')

fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, raw_prepro.epochs.info)

fwd_fixed = mne.convert_forward_solution(raw_prepro.fwr, surf_ori=True,
                                         force_fixed=True, use_cps=True)

inv = make_inverse_operator(raw_prepro.raw.info, fwd_fixed, noise_cov,
                            loose=0.2)

evoked = raw_prepro.epochs.average()
evoked.plot(time_unit='s')
evoked.plot_topomap(times=[-.5, 0, .4], time_unit='s')

# Show whitening
evoked.plot_white(noise_cov, time_unit='s')


stc = apply_inverse(evoked, inv, lambda2=1. / 9.)

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
