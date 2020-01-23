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
import mne
import os.path as op

from m_set_directories import set_paths


subject = '18011045C'  # example
# options 'Movie', 'CarTask', 'Flanker'
experiment = 'CarTask'

paths_dic = set_paths()


def get_subjects(experiment):
    # Get list of subjects with experiment recording
    pth_tmp = op.join(op.expanduser(paths_dic["root"]), '18011*', '*' +
                      experiment + '*')
    Subj_list = [op.dirname(x) for x in sorted(glob.glob(pth_tmp))]
    return Subj_list

# %%


def piepline(iSubj, option):
    ''' Function for running preprocessing steps '''

    o = option
    subject = op.basename(iSubj)
    print('Preprocessing subject: ' + subject)

    # %% Create Class object
    raw_prepro = MNEprepro(subject, experiment, paths_dic)

    # %% Detect and reject bad channels
    if o['bad_chns'][0] is True:
        raw_prepro.detect_bad_channels(zscore_v=4, overwrite=o['bad_chns'][1])
        if not raw_prepro.raw.info['bads'] == []:
            raw_prepro.raw.load_data().interpolate_bads(origin=[0, 0, 40])
    # %% Detect and reject moving periods
    if o['movement'][0] is True:
        raw_prepro.detect_movement(overwrite=o['movement'][1], plot=False)

    # %% Detect and reject periods with muscle
    if o['muscle'][0] is True:
        raw_prepro.detect_muscle(overwrite=o['muscle'][1], plot=True)

    # %%Run ICA
    try:
        if o['ICA_run'][0] is True:
            raw_prepro.run_ICA(overwrite=o['ICA_run'][1])
        if o['ICA_plot'][0] is True:
            raw_prepro.plot_ICA()
    except RuntimeError:
        return

    # %%Create epochs
    if o['epking'][0] is True:
        raw_prepro.epoching(overwrite=o['epking'][1], cond_name=o['epking'][2])

    # %%Create forward mddelling
    if o['src_model'][0] is True:
        raw_prepro.src_modelling(overwrite=o['src_model'][1])

    # %%Create noise cov for mne
    if o['mne_ncov'][0] is True:
        if o['epking'][2] is not None:  # "Here all cond epochs will be taken"
            raw_prepro.epoching(overwrite=o['epking'][1], cond_name=None)
        raw_prepro.mne_cov(overwrite=o['mne_ncov'][1])

    # %%Create mne inverse operator
    if o['mne_inv'][0] is True:
        raw_prepro.mne_inv_operator(overwrite=o['mne_inv'][1])

    return raw_prepro
# %%


def main():
    pass


if __name__ == "__main__":
    Car_task_cond = ['Transp/H2L', 'Transp/L2H', 'NotTransp/H2L', 'NotTransp/L2H']

    opt_run_overwrite = dict()
    opt_run_overwrite['bad_chns'] = [True, False]
    opt_run_overwrite['movement'] = [False, False]
    opt_run_overwrite['muscle'] = [True, False]
    opt_run_overwrite['ICA_run'] = [True, False]
    opt_run_overwrite['ICA_plot'] = [False, False]  # if clean no nned to run
    opt_run_overwrite['epking'] = [True, False, None]  # None=take all cond.
    opt_run_overwrite['src_model'] = [True, True]
    opt_run_overwrite['mne_ncov'] = [True, False]
    opt_run_overwrite['mne_inv'] = [True, True]

    Subj_list = get_subjects(experiment)

    # %% Make epochs with different conditions length
    for i in range(1, 4):
        opt_run_overwrite['epking'][2] = Car_task_cond[i]
        raw_prepro = [piepline(iSubj, opt_run_overwrite) for iSubj in Subj_list]

    # %% Epochs all together
    raw_prepro = [piepline(iSubj, opt_run_overwrite) for iSubj in Subj_list]

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
