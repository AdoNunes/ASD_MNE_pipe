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

RT = dict()
p = dict()
for ic in range(4):
    condition = Car_task_cond[ic]
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

    # preallocate & ensure index match subj
    n_subjects = len(subj_epo)
    evoked_all[c] = [None]*n_subjects
    RT[c] = [None]*n_subjects

    subj_ix = {}
    subj_ix['all'] = list(range(len(subj_epo)))
    subj_ix['ASD'] = [s for s in range(len(subj_epo)) if 'A' in op.basename(subj_epo[s])[8]]
    subj_ix['CTR'] = [s for s in range(len(subj_epo)) if 'C' in op.basename(subj_epo[s])[8]]

    # Get evoked data
    for ix, iSubj in enumerate(subj_epo):
        epochs = mne.read_epochs(iSubj)
        t_beg, t_resp, t_out, t_end = get_epoch_times(epochs, plot=False)
        RT[c][ix] = np.mean(np.abs([i - j for i, j in zip(t_resp, t_out)
                                  if type(i) is not list]))
    #    evoked_all[c].append(epochs.average())

    rt = dict()
    grp = ['all', 'ASD', 'CTR']
    for g in grp:
        gix = subj_ix[g]
        rt[g] = [[RT[c][i] for i in gix]][-1]
        print (c, np.average(rt[g])/600, g)
    
    p[c] = ttest_ind(rt['ASD'], rt['CTR'])


    evoked_grp = [evoked_all[c][i] for i in gix]
    a = mne.combine_evoked(evoked_grp[:], 'nave')
    a.plot_joint(title=g + ' ' + c)

t_beg, t_resp, t_out, t_end = get_epoch_times()


plt.figure(), plt.plot( epoch.times,epoch.get_data(pik)[:, 0, :].T)
plt.plot(epoch.times[PD_off],[7]*len(PD_off), 'x', linewidth=5)
