#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 09:36:21 2019

@author: an512
"""
import numpy as np
from m_set_directories import set_paths
from S02_prepro_loop import piepline, get_subjects

experiment = 'CarTask'

paths_dic = set_paths()
Subj_list = get_subjects(experiment)

opt_run_overwrite = dict()
opt_run_overwrite['bad_chns'] = [False, False]
opt_run_overwrite['movement'] = [False, False]
opt_run_overwrite['muscle'] = [False, False]
opt_run_overwrite['ICA_run'] = [False, False]
opt_run_overwrite['ICA_plot'] = [False, False]
opt_run_overwrite['epking'] = [False, False, None]  # None=take all conditions
opt_run_overwrite['src_model'] = [False, False]

# Get trial times for all
T_all_sbj = [None] * len(Subj_list)
for i, iSubj in enumerate(Subj_list):
    # Load class object, get events, store them
    raw_prepro = piepline(iSubj, opt_run_overwrite)
    raw_prepro.get_events()
    Time_events = raw_prepro.all_trl_info  # trigOn, PDon, PDoff, TID, PD-resp
    T_all_sbj[i] = Time_events[:, 3:5]

# Group indexing
Grp_ASD = np.array([s[-1] == 'A' for s in Subj_list])

# Compare conditions across groups
Cond_comb = [['H2L', 'L2H'], ['Transp', 'NotTransp']]

Conditions = raw_prepro.event_id

