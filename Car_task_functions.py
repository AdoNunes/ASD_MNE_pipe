#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 18:42:32 2019

@author: an512

Car task functions
"""


def get_epoch_times(epoch, PD_chn='UADC015-3007', trig_chn='UPPT001',
                    plot=False):
    """Get time start, response, out, end   """
    import numpy as np
    epoch.pick_types(misc=True, meg=False)

    fs = epoch.info['sfreq']
    pik = epoch.ch_names.index(PD_chn)
    n_trials =  epoch.get_data().shape[0]
    time_in = epoch.time_as_index(0)[0]
    PD = epoch.get_data(pik)[:, 0, time_in:]

    pik_tg = epoch.ch_names.index(trig_chn)
    Tg = epoch.get_data(pik_tg)[:, 0, :]
    resp_val = 1
    end_trl_val = 99
    beg_val = [10, 20, 30, 40]

    PD_OFF = [None] * n_trials
    time_beg = [None] * n_trials
    time_end = [None] * n_trials
    time_resp = [None] * n_trials
    time_out = [None] * n_trials
    for e in np.arange(n_trials):
        # Photodiode
        PD_OFF_temp = []
        percentile = np.percentile(PD[e, :], [1, 99])
        n, bins = np.histogram(PD[e, :], bins=50, range=percentile)
        T_PD = PD[e, :] > bins[26]  # set a treshold
        t_min = 0.16  # min PD length in ms
        min_samp4 = round(t_min * fs/4)  # quater PD min length
        min_samp8 = round(t_min * fs/8)  # 1/8 PD min length

        for ind, now in enumerate(T_PD):
            if (now == False and T_PD[ind-1] == True and
                np.all(T_PD[ind-min_samp8: ind-1] == True) and
                np.all(T_PD[ind+min_samp8: ind+min_samp4] == False)):
                PD_OFF_temp.append(ind)
        PD_OFF[e] = PD_OFF_temp[-1]
        # Response
        res_sam = [i for i, j in enumerate(Tg[e, :]) if j == resp_val]
        # Get one sample per response
        res = np.array([j for i, j in enumerate(res_sam)
                        if i == 0 or j - 1 > res_sam[i - 1]])

        resp_in = res[res > time_in]
        if resp_in.size is not 1:
            resp_in = [None]
        time_resp[e] = resp_in
        # Time end and begenning
        time_end_ix = np.where(Tg[e, :] == end_trl_val)
        time_end_ix_last = time_end_ix[-1]
        if time_end_ix_last.size > 0 and np.any(time_end_ix_last > time_in):
            t = time_end_ix_last[-1]
        elif time_resp[e] == [None]:  # if no end trigger and no resp
            t = PD_OFF[e] + time_in
        else:  # if no end trigger take either resp or PD off
            t = max(time_resp[e][0], PD_OFF[e] + time_in)
        time_end[e] = t
        X = [i for i in range(len(Tg[e, :])) if np.any(Tg[e, i] == beg_val)]
        if X == []:
            X = [0]
        else:
            time_beg[e] = X[0]
    # Get first PD OFF and add 0 time samples
    time_out = PD_OFF + time_in
    assert np.all(time_end >= time_out)

    if plot is True:
        import matplotlib.pyplot as plt
        plt.figure()
        for d in range(n_trials):
            rows = np.floor(np.sqrt(n_trials))
            cols = np.ceil(np.sqrt(n_trials))
            plt.subplot(rows, cols, 1+d)
            b = time_beg[d]
            e = time_end[d]
            plt.plot(epoch.times[b:e], Tg[d, b:e].T)
            plt.plot(epoch.times[b:e], epoch.get_data(pik)[d, 0, b:e].T)
            plt.plot(epoch.times[time_resp[d]], [7], 'go', linewidth=5)
            plt.plot(epoch.times[time_end[d]-1], [7], 'ro', linewidth=5)
            plt.plot(epoch.times[time_beg[d]], [7], 'bo', linewidth=5)
            plt.plot(epoch.times[time_out[d]], [7], 'ko', linewidth=5)
    return time_beg, time_resp, time_out, time_end

'''
plt.figure()
plt.plot(epoch.times, Tg.T )
plt.plot(epoch.times[time_beg],[7]*len(time_beg), 'bo', linewidth=5)
plt.plot(epoch.times[time_out],[7]*len(time_out), 'ro', linewidth=5)
plt.plot(epoch.times[time_resp],[7]*len(time_resp), 'go', linewidth=5)

plt.plot(epoch.times[time_resp],epoch.times[time_out], 'go', linewidth=5)


import matplotlib.pyplot as plt
'''
