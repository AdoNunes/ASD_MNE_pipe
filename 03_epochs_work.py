#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:16:00 2019

@author: an512
"""

import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from m_set_directories import set_paths
import glob
import os.path as op
import numpy as np
from Car_task import get_epoch_times

paths_dic = set_paths()
src_path = paths_dic['data2src']
experiment = 'CarTask'
Car_task_cond = ['Transp/H2L', 'Transp/L2H', 'NotTransp/H2L', 'NotTransp/L2H']


condition = Car_task_cond[1]
c = condition
evoked_all = dict()

evoked_all[c] = []

# Get path
if condition is not None:
    cond = condition.replace('/', '')
    outdir = f"{src_path}/*_{experiment}_{cond}-epo.fif"
else:
    outdir = f"{src_path}/*_{experiment}-epo.fif"

subj_epo = sorted(glob.glob(outdir))

crown = ['18011007', '18011008', '18011029', '18011032', '18011039']
subj_epo = [s for s in subj_epo if not any(ignore in s for ignore in crown)]

subj_ix = {}
subj_ix['all'] = list(range(len(subj_epo)))
subj_ix['ASD'] = [s for s in range(len(subj_epo)) if 'A' in op.basename(subj_epo[s])[8]]
subj_ix['CTR'] = [s for s in range(len(subj_epo)) if 'C' in op.basename(subj_epo[s])[8]]

RT = [];
# Get evoked data
for iSubj in subj_epo:
    epochs = mne.read_epochs(iSubj)
    t_beg, t_resp, t_out, t_end = get_epoch_times(epochs)
    RT.append(np.nanmean(np.abs([ i - j for i, j in zip(t_resp, t_out)])))
    evoked_all[c].append(epochs.average())

grp = ['all', 'ASD', 'CTR']
for g in grp:
    gix = subj_ix[g]

    evoked_grp = [evoked_all[c][i] for i in gix]
    a = mne.combine_evoked(evoked_grp[:], 'nave')
    a.plot_joint(title=g + ' ' + c)

t_beg, t_resp, t_out, t_end = get_epoch_times()


plt.figure(), plt.plot( epoch.times,epoch.get_data(pik)[:, 0, :].T)
plt.plot(epoch.times[PD_off],[7]*len(PD_off), 'x', linewidth=5)
