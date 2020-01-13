#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 09:36:21 2019

@author: an512
"""
import numpy as np
from pprint import pprint
from m_set_directories import set_paths
from S02_prepro_loop import piepline, get_subjects
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt


def plot_times(points, asd_grp, plt_title):
    "PLot time points for asd and ~asd, red dot for mean"
    plt.figure()
    plt.plot(asd_grp, points, 'o')
    plt.plot(1, np.average(points[asd_grp]), 'ro', LineWidth=5)
    plt.plot(0, np.average(points[~asd_grp]), 'ro', LineWidth=5)
    plt.title(plt_title)


experiment = 'CarTask'

paths_dic = set_paths()
Subj_list = get_subjects(experiment)
out = ['18011019A']
Subj_list = [s for s in Subj_list if not any(o in s for o in out)]

opt_run_overwrite = {'bad_chns': [False, False],
                     'movement': [False, False],
                     'muscle': [False, False],
                     'ICA_run': [False, False],
                     'ICA_plot': [False, False],
                     'epking': [False, False, None],  # [2]->Condition to take
                     'src_model': [False, False]}

# Get trial times for all
T_all_sbj = [None] * len(Subj_list)
for i, iSubj in enumerate(Subj_list):
    # Load class object, get events, store them
    raw_prepro = piepline(iSubj, opt_run_overwrite)
    raw_prepro.get_events()
    Time_events = raw_prepro.all_trl_info  # trigOn, PDon, PDoff, TID, PD-resp
    T_all_sbj[i] = Time_events[:, 3:5]


Fs = raw_prepro.raw.info['sfreq']
Conditions = raw_prepro.event_id

# Group indexing
Grp_ASD = np.array([s[-1] == 'A' for s in Subj_list])

# Count number of trials per condition
ntrials = {}
nt_arr = np.empty((0, len(T_all_sbj)))
for i, k in enumerate(Conditions.items()):
    k, v = k
    ntrials[k] = np.array([np.count_nonzero(s[:, 0] == v) for s in T_all_sbj])
    nt_arr = np.concatenate((nt_arr, [ntrials[k]]))
nt_arr = np.transpose(nt_arr)


####################
# Compare conditions between groups
Conditions = {'Transp/H2L': 10, 'Transp/L2H': 20,
              'NotTransp/H2L': 30, 'NotTransp/L2H': 40,
              'Transp': [10, 20], 'NotTransp': [30, 40]}

p = dict()
for k, v in Conditions.items():
    T_avg = np.array([
            np.nanmean(s[np.isin(s[:, 0], v), 1]) for s in T_all_sbj])
    p[k] = ttest_ind(T_avg[Grp_ASD], T_avg[~Grp_ASD])
pprint(p)

####################
# Compare conditions between groups RESP < 0
p_no0 = dict()
for k, v in Conditions.items():
    T_avg = np.array([
            s[np.isin(s[:, 0], v), 1] for s in T_all_sbj])
    T_avg = np.array([np.nanmean(s[s < 0]) for s in T_avg])
    p_no0[k] = ttest_ind(T_avg[Grp_ASD], T_avg[~Grp_ASD])
pprint(p_no0)

####################
# Diff within group between cond
p_grp_cond = dict()
gn = ['ctr', 'asd']

t = np.array([s[np.isin(s[:, 0], [30, 40]), 1] for s in T_all_sbj])
T1 = np.array([np.nanmean(s[s < 0]) for s in t])
t = np.array([s[np.isin(s[:, 0], [10, 20]), 1] for s in T_all_sbj])
T2 = np.array([np.nanmean(s[s < 0]) for s in t])
for i, g in enumerate([~Grp_ASD, Grp_ASD]):
    p_grp_cond[gn[i]] = ttest_ind(T1[g], T2[g])
    
plot_times(T1, Grp_ASD, 'Not_Trans')
plot_times(T2, Grp_ASD, 'Trans')

t_max = [np.nanmin(np.abs(s))/600 for s in T_all_sbj]


# PLot
for k, v in Conditions.items():
    T_avg = np.array([
            np.nanmean((s[s[:, 0] == v, 1])) for s in T_all_sbj
            ])/Fs
    plot_times(T_avg, Grp_ASD, k)

    plt.plot(1, np.average(T_avg[8]), 'mo', LineWidth=5)
    plt.plot(1, np.average(T_avg[9]), 'mo', LineWidth=5)
    plt.plot(1, np.average(T_avg[11]), 'go', LineWidth=5)
    plt.plot(0, np.average(T_avg[12]), 'ko', LineWidth=5)

####################
# Time corrected
T_avg1 = np.array([np.nanmean((s[s[:, 0] == 30, 1])) for s in T_all_sbj])
             - np.array([np.nanmean((s[s[:, 0] == 10, 1])) for s in T_all_sbj])
T_avg2 = np.array([np.nanmean((s[s[:, 0] == 40, 1])) for s in T_all_sbj])
           - np.array([np.nanmean((s[s[:, 0] == 20, 1])) for s in T_all_sbj])
T_avg3 = (T_avg1 + T_avg2)/2

p_correc = {'H2L': ttest_ind(T_avg1[Grp_ASD], T_avg1[~Grp_ASD])}
p_correc['L2H'] = ttest_ind(T_avg2[Grp_ASD], T_avg2[~Grp_ASD])
p_correc['All'] = ttest_ind(T_avg3[Grp_ASD], T_avg3[~Grp_ASD])

np.average(T_avg1[Grp_ASD])

plot_times(T_avg1/Fs, Grp_ASD, 'C0rrected')

####################
# Time corrected RESP <0

T_avg = []
for i, j in enumerate([10, 20, 30, 40]):
   t = [s[s[:,0]==j, 1] for s in T_all_sbj]
   T_avg.append( np.array([np.nanmean(s[s<0]) for s in t]))

T_corr1 =  np.array(T_avg[2] - T_avg[0])
T_corr2 =  np.array(T_avg[3] - T_avg[1])
T_corr3 =  (T_corr1 + T_corr2)/2

p_correc0 = {'H2L': ttest_ind(T_corr1[Grp_ASD], T_corr1[~Grp_ASD]),
            'L2H': ttest_ind(T_corr2[Grp_ASD], T_corr2[~Grp_ASD]),
            'All': ttest_ind(T_corr3[Grp_ASD], T_corr3[~Grp_ASD])}
plot_times(T_corr3/Fs, Grp_ASD, 'Corrected0')


###################
# Get right and wrongs
p_resp_corr = dict()
for k, v in Conditions.items():
    T_avg = np.array([s[np.isin(s[:, 0], v), 1] for s in T_all_sbj])
    T_avg2 = np.array([np.sum(s > 0) for s in T_avg])

    not_resp = np.array([np.sum(np.isnan(s)) for s in T_avg])
    T_avg_nonan = np.array([s[~np.isnan(s)] for s in T_avg])

    T_avg = np.array([np.sum(s > 0)/len(s) for s in T_avg_nonan])

    p_resp_corr[k] = ttest_ind(not_resp[Grp_ASD], not_resp[~Grp_ASD])
pprint(p_resp_corr)


