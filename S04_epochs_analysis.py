#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:16:00 2019

@author: an512
"""

from mne import (compute_covariance, convert_forward_solution, read_epochs,
                 combine_evoked, EvokedArray, read_forward_solution)
from mne.minimum_norm import make_inverse_operator, apply_inverse
from m_set_directories import set_paths
import glob
import os.path as op
import numpy as np
from Car_task import get_epoch_times
from scipy.stats import ttest_ind
from mne.stats import permutation_t_test
import matplotlib.pyplot as plt

# Set paths
paths_dic = set_paths()
src_path = paths_dic['data2src']
# Set cond and subj
experiment = 'CarTask'
Car_task_cond = ['Transp/H2L', 'Transp/L2H', 'NotTransp/H2L', 'NotTransp/L2H']
out_sbj = ['18011007', '18011008', '18011029', '18011032', '18011039']

Conds_2do = Car_task_cond


def get_subj_list_cond(Conds_2do, out_sbj):
    """ Gives dic with epoch path for each subject and condition 
    Conds_2do: list with condition names
    out_sbj: list str subjects to discard
    """
    subj_lst = dict()
    for c in Conds_2do:  # get subj epoch path for each condition str in list
        # Get path to subject epochs
        if c is not None:
            cond = c.replace('/', '')
            outdir = f"{src_path}/*_{experiment}_{cond}-epo.fif"
        else:
            outdir = f"{src_path}/*_{experiment}-epo.fif"
        subj_lst[c] = sorted(glob.glob(outdir))
        # Remove unwanted subjects
        subj_lst[c] = [s for s in subj_lst[c] if not any(o in s for o
                                                         in out_sbj)]
    return subj_lst


def get_stc(epoch_data):
    """Returns stc for a given `epoch_data`, assumes `fwr` path from 
    `epoch_data.filename` basename[0:9]+ fwd.fif """

    fwd_name_dir = op.dirname(epoch_data.filename)
    fwd_name_base = op.basename(epoch_data.filename)[0:9] + '_oct5-fwd.fif'
    fwd_name = op.join(fwd_name_dir, fwd_name_base)
    fwd = read_forward_solution(fwd_name)

    noise_cov = compute_covariance(epoch_data, tmax=0, method='shrunk')
    fwd_fixed = convert_forward_solution(fwd, surf_ori=True,
                                         force_fixed=True, use_cps=True)
    inv = make_inverse_operator(epoch_data.info, fwd_fixed, noise_cov,
                                loose=0.2)

    evoked = epoch_data.epochs.average()
    stc = apply_inverse(evoked, inv, lambda2=1. / 9.)


fwd_name = op.join(src_path, '18011003A' +'_oct5-fwd.fif')

fwd = read_forward_solution(fwd_name)

subj_lst = get_subj_list_cond(Conds_2do, out_sbj)

########################################
# Count number of trials per condition
ntrials = []
for ic, x in enumerate(Conds_2do):
    ntrials.append([])
    for ix, iSubj in enumerate(subj_lst[x]):
        epochs = read_epochs(iSubj, preload=False)
        ntrials[ic]. append(epochs.events.shape[0])

ntrials = np.vstack(np.array(ntrials)).T

########################################
# Get RT and stats for conditions
RT = dict()
p = dict()
for c in Conds_2do:
    evoked_all = dict()
    evoked_all[c] = []

    # preallocate & ensure index match subj
    n_subjects = len(subj_lst[c])
    evoked_all[c] = [None]*n_subjects
    RT[c] = [None]*n_subjects

    subj_ix = {}
    subj_ix['all'] = list(range(len(subj_lst[c])))
    subj_ix['ASD'] = [s for s in range(len(subj_lst[c]))
                      if 'A' in op.basename(subj_lst[c][s])[8]]
    subj_ix['CTR'] = [s for s in range(len(subj_lst[c]))
                      if 'C' in op.basename(subj_lst[c][s])[8]]
    # Get evoked data
    for ix, iSubj in enumerate(subj_lst[c]):
        epochs = read_epochs(iSubj)
        t_beg, t_resp, t_out, t_end = get_epoch_times(epochs, plot=False)
        RT[c][ix] = np.mean(np.abs([i - j for i, j in zip(t_resp, t_out)
                                    if type(i) is not list]))
    # Test diff
    rt = dict()
    grp = ['all', 'ASD', 'CTR']
    for g in grp:
        gix = subj_ix[g]
        rt[g] = [[RT[c][i] for i in gix]][-1]
        print(c, np.average(rt[g])/600, g)
    p[c] = ttest_ind(rt['ASD'], rt['CTR'])


########################################
# Get evoked for conditions
evoked_all = dict()
for c in Conds_2do:
    evoked_all[c] = []

    # preallocate & ensure index match subj
    n_subjects = len(subj_lst[c])
    evoked_all[c] = [None]*n_subjects

    subj_ix = {}
    subj_ix['all'] = list(range(len(subj_lst[c])))
    subj_ix['ASD'] = [s for s in range(len(subj_lst[c]))
                      if 'A' in op.basename(subj_lst[c][s])[8]]
    subj_ix['CTR'] = [s for s in range(len(subj_lst[c]))
                      if 'C' in op.basename(subj_lst[c][s])[8]]

    for ix, iSubj in enumerate(subj_lst[c]):
        epochs = read_epochs(iSubj)
        evoked = epochs.average()
        evoked_all[c][ix] = epochs.average()

    evoked_grp = [evoked_all[c][i] for i in gix]
    a = combine_evoked(evoked_grp, 'nave')
    a.plot_joint(title=g + ' ' + c)

########################################
# Compare group differences

windows = np.array([[-.4, -.2], [-.2, 0], [0, .15], [.2, .5]])
nwins = windows.shape[0]
stats = {}
for w in windows:
    for c in Conds_2do:
        n_subjects = len(evoked_all[c])
        stats[c] = np.array([])
        epok_win = []

        for ix, iSubj in enumerate(evoked_all[c]):
            evo = iSubj.copy().crop(w[0], w[1]).data
            evo = np.average(evo, axis=1)
            epok_win.append(evo)

        s_grp = {}
        n_sens = epok_win[0].shape[0]
        for s in range(epok_win[0].shape[0]):
            s_list = [x[s] for x in epok_win]
            grp = ['ASD', 'CTR']
            for g in grp:
                gix = subj_ix[g]
                s_grp[g] = [s_list[i] for i in gix]
            test, pv = ttest_ind(s_grp['ASD'], s_grp['CTR'])
            stats[c] = np.append(stats[c], pv)
        plt.figure(), plt.plot(stats[c]), plt.title([c, w]), plt.axhline(y=.05)
        stats[c] = np.array(stats[c])
        # Plot
        mask = stats[c][:, np.newaxis] <= 0.05
        evoked = EvokedArray(stats[c][:, np.newaxis], iSubj.info, tmin=0.)
        evoked.plot_topomap(title=[c, w], units='p', vmin=0., vmax=.05,
                            times=[0], scalings=1, time_format=None, mask=mask,
                            cbar_fmt='-%0.2f')










