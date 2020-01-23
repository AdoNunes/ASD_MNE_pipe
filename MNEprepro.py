
from os import path as op
from scipy.stats import zscore
import glob
import mne
import numpy as np
import csv
from mne.annotations import Annotations, read_annotations
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure as pfig
from matplotlib.pyplot import  plot as pplot
from scipy import stats
import pandas as pd
from scipy.ndimage.measurements import label
from mne.io.ctf.trans import _quaternion_align
from mne.chpi import _apply_quat
import math


class MNEprepro():

    """
    Class to preproces CTF data
    Usage:
    raw_prepro = MNEprepro(subject, experiment, paths_dic)

    paths_dic = {
        "root": "~/Desktop/projects/MNE/data",
        "subj_anat": 'anatomy'
        "out": "~/Desktop/projects/MNE/data_prep"
    }

    subject = '18011014C'
    experiment = 'Movie'

    root = folder where subjects name folders are
    subj_anat = name anatomy folder in subject folder
    out = path to output files

    FOLDER EXAMPLE:
    /path2/MEG_data:
                    /subject_name
                                 /experiment1.ds
                                 /experiment2.ds
                                 /anatomy
                                         /t1.nii.gz
                                         /cortical_surf_4k.gii
                                /polhemus.pos


    STEPS to do

    1- load the data
    2- Load previous prepro info -> bad channels, mov, etc if any
    3- plot and find bad channels
    4- movement rejection
    5- muscle rejection
    6- ICA or SSP
    7- epoch (save or no save)
    7a- epoch based on photodiode
    8- head model
    8- do sensor level:
        Pow
        ERF
        Conn
    9- source level: ...



    out directory:
        /bad_chn_annot
        /ICA
    """

    def __init__(self, subject, experiment, paths_dic):
        self.subject = subject
        self.experiment = experiment
        self.pth_root = op.expanduser(paths_dic["root"])
        self.pth_out = op.expanduser(paths_dic["out"])
        self.pth_FS = op.expanduser(paths_dic["FS"])
        mne.set_config('SUBJECTS_DIR', self.pth_FS)
        self.check_outdir()
        self.pth_subject = op.join(self.pth_root, subject)
        if self.check_MXfilter_sbjs() is True:
            print('Using MaxFilter preprocessed data')
            self.raw = mne.io.read_raw_fif(self.pth_raw, preload=False)
        else:
            self.pth_raw = glob.glob(op.join(self.pth_subject, subject) + '_'
                                     + experiment + '*')[-1]
            self.raw = mne.io.read_raw_ctf(self.pth_raw, preload=False)
            if self.raw.compensation_grade != 3:
                self.raw.apply_gradient_compensation(3)

    def check_outdir(self):
        from os import makedirs
        out_dir = self.pth_out
        self.out_bd_ch = op.join(out_dir, 'bad_chans')
        self.out_annot = op.join(out_dir, 'annots')
        self.out_ICAs = op.join(out_dir, 'ICAs')
        self.out_srcData = op.join(out_dir, 'data2src')
        makedirs(self.out_bd_ch, exist_ok=True)  # I want to loop it
        makedirs(self.out_annot, exist_ok=True)
        makedirs(self.out_ICAs, exist_ok=True)
        makedirs(self.out_srcData, exist_ok=True)

    def check_MXfilter_sbjs(self):
        pth_mx_sbj = self.pth_out + '/MX_filter_subj/' + self.subject
        if op.exists(pth_mx_sbj):
            self.pth_raw = glob.glob(op.join(pth_mx_sbj, self.subject +
                                             '_' + self.experiment + '*'))[-1]
            is_MX = True
        else:
            is_MX = False
        return is_MX

    def detect_bad_channels(self, zscore_v=4, overwrite=False, method='both',
                            neigh_max_distance=.035):
        """ zscore_v = zscore threshold, save_csv: path_tosaveCSV"""
        fname = self.subject + '_' + self.experiment + '_bads.csv'
        out_csv_f = op.join(self.out_bd_ch, fname)
        if op.exists(out_csv_f) and not overwrite:
            bad_chns = csv_read(out_csv_f)
            print('Reading from file, bad chans are:', bad_chns)
        else:
            from itertools import compress
            print('Looking for bad channels')

            # set recording length
            Fs = self.raw.info['sfreq']
            t1x = 30
            t2x = 220
            t2 = min(self.raw.last_samp/Fs, t2x)
            t1 = max(0, t1x + t2-t2x)  # Start earlier if recording is shorter

            # Get data
            raw_copy = self.raw.copy().crop(t1, t2).load_data()
            raw_copy = raw_copy.pick_types(meg=True, ref_meg=False)\
                .filter(1, 45).resample(150, npad='auto')
            data_chans = raw_copy.get_data()

            # Get channel distances matrix
            chns_locs = np.asarray([x['loc'][:3] for x in
                                    raw_copy.info['chs']])
            chns_dist = np.linalg.norm(chns_locs - chns_locs[:, None],
                                       axis=-1)
            chns_dist[chns_dist > neigh_max_distance] = 0

            # Get avg channel uncorrelation between neighbours
            chns_corr = np.abs(np.corrcoef(data_chans))
            weig = np.array(chns_dist, dtype=bool)
            chn_nei_corr = np.average(chns_corr, axis=1, weights=weig)
            chn_nei_uncorr_z = zscore(1-chn_nei_corr)  # l ower corr higer Z

            # Get channel magnitudes
            max_Pow = np.sqrt(np.sum(data_chans ** 2, axis=1))
            max_Z = zscore(max_Pow)

            if method == 'corr':  # Based on local uncorrelation
                feat_vec = chn_nei_uncorr_z
                max_th = feat_vec > zscore_v
            elif method == 'norm':  # Based on magnitude
                feat_vec = max_Z
                max_th = feat_vec > zscore_v
            elif method == 'both':  # Combine uncorrelation with magnitude
                feat_vec = (chn_nei_uncorr_z+max_Z)/2
                max_th = (feat_vec) > zscore_v

            bad_chns = list(compress(raw_copy.info['ch_names'], max_th))
            raw_copy.info['bads'] = bad_chns
            if bad_chns:
                print(['Plotting data,bad chans are:'] + bad_chns)
                pfig(), pplot(feat_vec), plt.axhline(zscore_v)
                plt.title(['Plotting data,bad chans are:'] + bad_chns)
                raw_copy.plot(n_channels=100, block=True, bad_color='r')
                bad_chns = raw_copy.info['bads']
                print('Bad chans are:', bad_chns)
            else:
                print('No bad chans found')
            csv_save(bad_chns, out_csv_f)
            self.ch_max_Z = max_Z
        self.raw.info['bads'] = bad_chns

    def detect_movement(self, thr_mov=.01, plot=True, overwrite=False,
                        save=True):
        from mne.transforms import read_trans
        fname = self.subject + '_' + self.experiment + '_mov.txt'
        out_csv_f = op.join(self.out_annot, fname)
        fname_t = self.subject + '_' + self.experiment + '_dev2head-trans.fif'
        out_csv_f_t = op.join(self.out_annot, fname_t)
        if op.exists(out_csv_f) and not overwrite:
            mov_annot = read_annotations(out_csv_f)
            print('Reading from file, mov segments are:', mov_annot)
            print('Reading from file, dev to head transformation')
            dev_head_t = read_trans(out_csv_f_t)
        else:
            print('Calculating head pos')
            pos = mne.chpi._calculate_head_pos_ctf(self.raw, gof_limit=-1)
            mov_annot, hpi_disp, dev_head_t = annotate_motion(self.raw, pos,
                                                              thr=thr_mov)
            if plot is True:
                plt.figure()
                plt.plot(hpi_disp)
                plt.axhline(y=thr_mov, color='r')
                plt.show(block=True)
            if save is True:
                mov_annot.save(out_csv_f)
                dev_head_t.save(out_csv_f_t)
            #fig.savefig(out_csv_f[:-4]+'.png')
        old_annot = self.raw.annotations #  if orig_time cant + with none time
        self.raw.set_annotations(mov_annot)
        self.raw.set_annotations(self.raw.annotations + old_annot)
        self.raw.info['dev_head_t_old'] = self.raw.info['dev_head_t']
        self.raw.info['dev_head_t'] = dev_head_t
        self.annot_movement = mov_annot

    def detect_muscle(self, thr=1.5, t_min=.5, plot=True, overwrite=False):
        """Find and annotate mucsle artifacts - by Luke Bloy"""
        fname = self.subject + '_' + self.experiment + '_mus.txt'
        out_csv_f = op.join(self.out_annot, fname)
        if op.exists(out_csv_f) and not overwrite:
            mus_annot = read_annotations(out_csv_f)
            print('Reading from file, muscle segments are:', mus_annot)
        else:
            print('Calculating muscle artifacts')
            raw = self.raw.copy().load_data()
            raw.pick_types(meg=True, ref_meg=False)
            raw.notch_filter(np.arange(60, 241, 60), fir_design='firwin')
            raw.filter(110, 140, fir_design='firwin')
            raw.apply_hilbert(envelope=True)
            sfreq = raw.info['sfreq']
            art_scores = stats.zscore(raw._data, axis=1)
            # band pass filter the data
            art_scores_filt = mne.filter.filter_data(art_scores.mean(axis=0),
                                                     sfreq, None, 4)
            art_mask = art_scores_filt > thr
            # remove artifact free periods shorter than t_min
            idx_min = t_min * sfreq
            comps, num_comps = label(art_mask == 0)
            for l in range(1, num_comps+1):
                l_idx = np.nonzero(comps == l)[0]
                if len(l_idx) < idx_min:
                    art_mask[l_idx] = True
            mus_annot = _annotations_from_mask(raw.times, art_mask,
                                               'Bad-muscle')
            if plot:
                del raw
                print('Plotting data, mark or delete art, by pressing a \n'
                      'Marked or demarked channels will be saved')
                old_bd_chns = self.raw.info['bads']
                raw = self.raw.copy().load_data().pick_types(meg=True,
                                                             ref_meg=False)
                raw.notch_filter(np.arange(60, 181, 60), fir_design='firwin')
                raw.filter(1, 140)
                old_annot = raw.annotations #  if orig_time cant + none
                raw.set_annotations(mus_annot)
                raw.set_annotations(raw.annotations + old_annot)
                raw.plot(n_channels=140, block=True, bad_color='r')
                mus_annot = raw.annotations
                if not (old_bd_chns == raw.info['bads']):
                    bad_chns = raw.info['bads']
                    print('Saving new bad channels list \n ')
                    print('Bad chans are:', bad_chns)
                    fname = self.subject + '_' + self.experiment + '_bads.csv'
                    csv_save(bad_chns, op.join(self.out_bd_ch, fname))
            mus_annot.save(out_csv_f)
        old_annot = self.raw.annotations #  if orig_time cant + with none time
        self.raw.set_annotations(mus_annot)
        self.raw.set_annotations(self.raw.annotations + old_annot)
        
        self.annot_muscle = mus_annot

    def run_ICA(self, overwrite=False):
        fname = self.subject + '_' + self.experiment + '-ica.fif.gz'
        out_fname = op.join(self.out_ICAs, fname)
        if op.exists(out_fname) and not overwrite:
            self.ica = mne.preprocessing.read_ica(out_fname)
        else:
            from mne.preprocessing import ICA
            raw_copy = self.raw.copy().load_data().filter(1, 45)
            self.ica = ICA(method='fastica', random_state=42,
                           n_components=0.99, max_iter=1000)
            picks = mne.pick_types(raw_copy.info, meg=True, ref_meg=False,
                                   stim=False, exclude='bads')
            reject = dict(grad=4000e-13, mag=6e-12)  # what rejec intervals?
            self.ica.fit(raw_copy, picks=picks, reject=reject, decim=3)
            self.ica.detect_artifacts(raw_copy)
            self.ica.done = False
            self.ica.save(out_fname)

    def plot_ICA(self, check_if_done=True, overwrite=False):
        fname = self.subject + '_' + self.experiment + '-ica.fif.gz'
        out_fname = op.join(self.out_ICAs, fname)
        # Load previous ICA instance
        if op.exists(out_fname) and not overwrite:
            self.ica = mne.preprocessing.read_ica(out_fname)
        else:
            # self.run_ICA(self)
            return
        # Check if ICA comps were inspected
        data_not_clean = True
        if check_if_done is True:
            if self.ica.info['description'] == 'done':
                data_not_clean = False
        # Plot interactively to select bad comps
        if data_not_clean is True:
            raw_copy = self.raw.copy().load_data().filter(1, 45)
        while data_not_clean is True:
            # ICA comp plotting
            self.ica.plot_components(inst=raw_copy)
            self.ica.plot_sources(raw_copy, block=True)
            # Clean and raw sensor plotting

            raw_plot = raw_copy.copy().pick_types(meg=True, ref_meg=False)
            raw_plot.plot(n_channels=80, title='NO ICA')

            raw_ica = raw_copy.copy().pick_types(meg=True, ref_meg=False)
            self.ica.apply(raw_ica)
            raw_ica.plot(n_channels=80, title='ICA cleaned', block=True)
            data_not_clean = bool(int(input("Select other ICA components? "
                                            "[0-no, 1-yes]: ")))
            if data_not_clean is False:
                self.ica.info['description'] = 'done'
                self.ica.save(out_fname)
        #self.ica.apply(self.raw.load_data())

    def get_events(self, plot=False, movie_annot=None):
        # general description of the data
        raw_copy = self.raw.copy()
        task = self.experiment
        fs = raw_copy.info['sfreq']
        time = raw_copy.buffer_size_sec
        N_samples = raw_copy.n_times
        ID = self.subject
        print('Data for subject ' + ID + ' is composed from')
        print(str(N_samples) + ' samples with ' + str(fs) + ' Hz sampl rate')
        print('Total time = ' + str(time) + 's')
        # get photodiode events from CTF data
        PD_ts, Ind_PD_ON, Ind_PD_OFF, T_PD = get_photodiode_events(raw_copy,
                                                                   fs)
        # pick Trigger channel time series from CTF data
        Trig = mne.io.pick.pick_channels_regexp(raw_copy.info['ch_names'],
                                                'UPPT001')
        Trig_ts = raw_copy.get_data(picks=Trig)
        # get events from trigger channel
        events_trig = mne.find_events(raw_copy, stim_channel='UPPT001',
                                      shortest_event=1)

        print(str(len(Ind_PD_ON)) + ' PD ONSETS FOUND')

        if task == 'CarTask':
            event_id = {'Transp/H2L': 10, 'Transp/L2H': 20,
                        'NotTransp/H2L': 30, 'NotTransp/L2H': 40}
            # get trigger names for PD ON states
            events = get_triger_names_PD(event_id, Ind_PD_ON, events_trig)
            all_trl_info, col_info = get_all_trl_info(event_id, Ind_PD_ON,
                                                      Ind_PD_OFF, events_trig)
            self.all_trl_info = all_trl_info
            self.all_trl_info_col_names = col_info
            # get different event length for each condition
            event_len = {'Transp/H2L': [-.86, 2.42],
                         'Transp/L2H': [-3.17, 2.3],
                         'NotTransp/H2L': [-.9, 2.37],
                         'NotTransp/L2H': [-3.17, 2.1]}
            self.event_len = event_len

        elif task == 'Movie':
            if movie_annot is not None:
                if fs != 600:
                    raise ValueError('Sampling must be 600Hz')
                event_id, events = get_pd_annotations(Ind_PD_ON, events_trig,
                                                      movie_annot)
            else:
                event_id = {'SceneOnset': 1}
                events = np.zeros((len(Ind_PD_ON), 3))
                events[:, 0] = Ind_PD_ON
                events[:, 2] = 1
            all_trl_info = None

        elif task == 'Flanker':
            event_id = {'Incongruent/Left': 3, 'Incongruent/Right': 5,
                        'Congruent/Left': 4, 'Congruent/Right': 6,
                        'Reward/Face': 7, 'Reward/Coin': 8,
                        'Reward/Neut_FaceTRL': 9, 'Reward/Neut_CoinTRL': 10}
            # get trigger names for PD ON states
            events = get_triger_names_PD(event_id, Ind_PD_ON, events_trig)
            all_trl_info = None

        events = events.astype(np.int64)
        # plot trig and PD
        if plot:
            plt.figure()
            plot_events(PD_ts, Ind_PD_ON, T_PD, Ind_PD_OFF, Trig_ts, events,
                        task, ID, all_trl_info=all_trl_info)
        self.event_id = event_id
        self.events = events

    def epoching(self, tmin=-0.5, tmax=0.5, plot=False, f_min=1, f_max=45,
                 overwrite=False, apply_ica=True, cond_name=None,
                 movie_annot=None, save=True):
        if cond_name is not None:
            fname = "%s_%s_%s-epo.fif" % (self.subject, self.experiment,
                                          cond_name.replace('/', ''))
        else:
            fname = "%s_%s-epo.fif" % (self.subject, self.experiment)
        out_fname = self.out_srcData + '/' + fname

        if op.exists(out_fname) and not overwrite:
            print('Reading epoched data from file')
            self.epochs = mne.read_epochs(out_fname)

        else:
            self.get_events(plot, movie_annot)
            raw_copy = self.raw.copy().load_data()
            info = raw_copy.info
            picks = mne.pick_types(info, meg=True, ref_meg=False)
            stim_ch = ['UADC015-3007', 'UPPT001']
            pick_stim = mne.pick_channels(info['ch_names'], stim_ch)
            pick_epo = np.append(picks, pick_stim)
            raw_copy.pick(pick_epo)

            pick_fil = mne.pick_types(raw_copy.info, meg=True, ref_meg=False)
            raw_copy.filter(f_min, f_max, picks=pick_fil)
            if cond_name is not None:  # Different conds diff lengths
                event = self.events
                ids = self.event_id[cond_name]
                tmin, tmax = self.event_len[cond_name]
                events = event[event[:, 2] == ids]
                epochs = mne.Epochs(raw_copy, events=events, tmin=tmin,
                                    tmax=tmax, event_id={cond_name: ids},
                                    baseline=(tmin, 0.0)).load_data()
            else:
                epochs = mne.Epochs(raw_copy, events=self.events, tmin=tmin,
                                    tmax=tmax, event_id=self.event_id,
                                    baseline=(tmin, 0.0)).load_data()
            self.epochs = epochs
            if apply_ica is True:
                if hasattr(self, 'ica'):  # Do ICA only on meg chns
                    piks = mne.pick_types(epochs.info, meg=True)
                    epochs_data = epochs.copy().pick(piks)
                    self.ica.apply(epochs_data)
                    self.epochs._data[:, piks, :] = epochs_data._data
                else:
                    return
            self.epochs.save(out_fname, overwrite=overwrite)

    def src_modelling(self, spacing=['oct5'], overwrite=False):
        from mne import (read_forward_solution, make_forward_solution,
                         write_forward_solution, setup_source_space)
        subject = self.subject
        mne.set_config('SUBJECTS_DIR', self.pth_FS)
        FS_subj = op.join(self.pth_FS, subject)
        fname_trans = op.join(FS_subj, subject + '-trans.fif')
        fname_bem = op.join(FS_subj, '%s-bem_sol.fif' % subject)

        if not op.exists(fname_bem) or overwrite:
            mne.bem.make_watershed_bem(subject, overwrite=True,
                                       volume='T1', atlas=True, gcaatlas=False,
                                       preflood=None)

            model = mne.make_bem_model(subject, ico=4, conductivity=(0.3,))
            bem = mne.make_bem_solution(model)
            mne.write_bem_solution(fname_bem, bem)
        else:
            bem = mne.read_bem_solution(fname_bem)

        for space in spacing:
            fname_src = op.join(FS_subj, 'bem', '%s-src.fif' % space)
            bname_fwd = '%s_%s-fwd.fif' % (subject, space)
            fname_fwd = op.join(self.out_srcData, bname_fwd)
            if not op.exists(fname_src) or overwrite:
                src = setup_source_space(subject, space,
                                         subjects_dir=self.pth_FS)
                src.save(fname_src, overwrite=overwrite)

            if op.exists(fname_fwd) and not overwrite:
                self.fwd = read_forward_solution(fname_fwd)
            else:
                self.fwd = make_forward_solution(self.raw.info, fname_trans,
                                                 fname_src, fname_bem)
                write_forward_solution(fname_fwd, self.fwd, overwrite)

    def mne_cov(self, overwrite=False):
        from mne import compute_covariance, read_cov
        fname = self.subject + '_' + self.experiment + '_ncov-cov.fif.gz'
        out_fname = op.join(self.out_srcData, fname)
        if op.exists(out_fname) and not overwrite:
            print('Reading noise covariance from file')
            self.ncov = read_cov(out_fname)
        else:
            self.ncov = compute_covariance(self.epochs, tmax=0, method='shrunk')
            self.ncov.save(out_fname)

    def mne_inv_operator(self, overwrite=False):
        from mne.minimum_norm import (read_inverse_operator,
                                      make_inverse_operator,
                                      write_inverse_operator)
        fname = self.subject + '_' + self.experiment + '_mne-inv.fif.gz'
        out_fname = op.join(self.out_srcData, fname)
        if op.exists(out_fname) and not overwrite:
            print('Reading inverse operator from file')
            self.inv = read_inverse_operator(out_fname)
        else:
            self.inv = make_inverse_operator(self.epochs.info, self.fwd,
                                             self.ncov, loose=0.2, depth=0.8)
            write_inverse_operator(out_fname, self.inv)


##################################################################
# Photod Diode functions
##################################################################


def get_photodiode_events(raw, fs, plot=False):
    raw = raw.copy()
    # pick PD channel time series from CTF data
    raw.info['comps'] = []
    PD = mne.io.pick.pick_channels_regexp(raw.info['ch_names'], 'UADC015-3007')
    PD_ts = raw.get_data(picks=PD)

    # find all 'ON' states of PD
    if plot:
        n, bins, patches = plt.hist(PD_ts[0, :], bins=50,
                                    range=(np.percentile(PD_ts[0, :], 1),
                                    np.percentile(PD_ts[0, :], 99)),
                                    color='#0504aa', alpha=0.7, rwidth=0.85)
    n, bins = np.histogram(PD_ts[0, :], bins=50,
                           range=(np.percentile(PD_ts[0, :], 1),
                                  np.percentile(PD_ts[0, :], 99)))
    thr = bins[26]  # set a treshold
    T_PD = PD_ts > thr
    Ind_PD_ON = []
    Ind_PD_OFF = []
    t_min = 0.16  # min PD length in ms
    min_samp4 = round(t_min * fs/4)  # quater PD min length
    min_samp8 = round(t_min * fs/8)  # 1/8 PD min length
    for ind, n in enumerate(T_PD[0, :]):
        if (n == True and T_PD[0, ind-1] == False and
            np.all(T_PD[0, ind-min_samp8:ind-1] == False) and
            np.all(T_PD[0, ind+min_samp8:ind+min_samp4] == True)):
            Ind_PD_ON.append(ind)
        elif (n == False and T_PD[0, ind-1] == True and
              np.all(T_PD[0, ind-min_samp8:ind-1] == True) and
              np.all(T_PD[0, ind+min_samp8:ind+min_samp4] == False) and ind>0):
            Ind_PD_OFF.append(ind)
    # PD on and PD off have to be the same length
    if len(Ind_PD_ON) < len(Ind_PD_OFF):
        Ind_PD_OFF_cor = []
        for t in Ind_PD_ON:
            t1 = (np.array(Ind_PD_OFF)-t)
            m = min(i for i in t1 if i > 0)
            t2 = list(np.where(t1 == m))
            Ind_PD_OFF_cor.append(Ind_PD_OFF[t2[0][0]])
        Ind_PD_OFF = Ind_PD_OFF_cor
    elif len(Ind_PD_ON) > len(Ind_PD_OFF):
        Ind_PD_ON_cor = []
        for t in Ind_PD_OFF:
            t1 = (np.array(Ind_PD_ON)-t) * -1
            m = min(i for i in t1 if i > 0)
            t2 = list(np.where(t1 == m))
            Ind_PD_ON_cor.append(Ind_PD_ON[t2[0][0]])
        Ind_PD_ON = Ind_PD_ON_cor
    return PD_ts, Ind_PD_ON, Ind_PD_OFF, T_PD


def plot_events(PD_ts, Ind_PD_ON, T_PD, Ind_PD_OFF, Trig_ts, events, task, ID,
                all_trl_info=None):

    plt.plot((PD_ts[0, :]))
    plt.plot(T_PD[0, :]*5)
    y = np.ones((1, len(Ind_PD_ON)))
    y = 0.5*y
    plt.plot(np.array(Ind_PD_ON), y[0, :], 'o', color='black')
    y = np.ones((1, len(Ind_PD_OFF)))
    y = 0.5*y
    plt.plot(np.array(Ind_PD_OFF), y[0, :], 'o', color='red')
    plt.plot(events[:, 0], events[:, 2], 'o', color='green')
    plt.plot((Trig_ts[0, :]))
    if task == 'CarTask':
        resp = []
        for ind, n in enumerate(all_trl_info[:, 4]):
            if not math.isnan(n):
                resp.append(n+all_trl_info[ind, 2])
        y = np.ones((1, len(resp)))
        y = 4*y
        plt.plot(resp, y[0, :], 'o', color='blue')
    plt.ylabel('a.u.')
    plt.xlabel('samples')
    plt.title('PD and Trigger events timing for ' + ID + ' ' + task + ' task')


def get_triger_names_PD(event_id, Ind_PD_ON, events_trig):
    # create events
    events = np.zeros((len(Ind_PD_ON), 3))
    events[:, 0] = Ind_PD_ON
    # get trigger names for PD ON states
    for key, value in event_id.items():
        ind = events_trig[:, 2] == value
        time = events_trig[ind, 0]
        for n in time:
            inx = (Ind_PD_ON-n)
            if np.any(inx[inx > 0]):
                m = min(inx[inx > 0])
                events[inx == m, 2] = value
    return events


def get_all_trl_info(event_id, Ind_PD_ON, Ind_PD_OFF, events_trig):
    # create additional object (1st col - trig start, 2nd col - PDon, 3rd col -
    # PDoff, 4th col - trig ID, 5th col - PDoff-Resp)
    all_trl_info_col_names = ['Trig_start', 'PD_on', 'PD_off', 'Trig_ID',
                              'PDoff-Resp']
    Trig_PDon_off = np.zeros((len(Ind_PD_ON), 5))
    Trig_PDon_off[:, 1] = Ind_PD_ON
    Trig_PDon_off[:, 2] = Ind_PD_OFF
    # get trigger names for PD ON states
    for key, value in event_id.items():
        ind = events_trig[:, 2] == value
        time = events_trig[ind, 0]
        for n in time:
            inx = (Ind_PD_ON-n)
            if np.any(inx[inx > 0]):
                m = min(inx[inx > 0])
                # events[inx == m, 2] = value
                Trig_PDon_off[inx == m, 3] = value
                Trig_PDon_off[inx == m, 0] = n
    # compute response time (PDoff - resp)
    all_trl_info = get_response(Trig_PDon_off, events_trig)
    return all_trl_info, all_trl_info_col_names


def get_response(Trig_PDon_off, events_trig):
    # compute time of response compare to PD offset

    all_trl_info = Trig_PDon_off
    for ind, n in enumerate(Trig_PDon_off):
        # find trigger value 1 in between PD on and next trial trigger
        Tunnel_in = Trig_PDon_off[ind, 1]
        if ind+1 == len(Trig_PDon_off):
            Next_trial = Tunnel_in + 900
        else:
            Next_trial = Trig_PDon_off[ind+1, 0]
        resp = []
        for res_ind in events_trig:
            if (res_ind[2] == 1 and res_ind[0] > Tunnel_in and res_ind[0]
                < Next_trial):
                resp.append(res_ind[0])
        # if there is more than one trig 1 (multiple button presses) - put NaN
        if len(resp) == 1:
            all_trl_info[ind, 4] = resp - Trig_PDon_off[ind, 2]
        else:
            all_trl_info[ind, 4] = float('nan')
    return all_trl_info


def get_pd_annotations(Ind_PD_ON, events_trig, movie_annot):
    # creates event id and event file based on movie annotation
    # find 1st PD of the 2nd movie presentation
    n = np.where(events_trig[:, 2] == 4)
    n = n[0][-1]
    trig_time = events_trig[n, 0]
    diff_time = Ind_PD_ON - trig_time
    pd_val = min(pp for pp in diff_time if pp > 0)
    pd_ind = np.where(diff_time == pd_val)
    PD = [Ind_PD_ON[0], Ind_PD_ON[pd_ind[0][0]]]
    i = 0
    cond = ['faces', 'facesNO', 'grasp', 'graspNO']
    for PD_i in PD:
        for c in cond:
            events_tmp = np.zeros((len(movie_annot[c][0, :]), 3))
            events_tmp[:, 0] = movie_annot[c][0, :] + PD_i
            events_tmp[:, 2] = movie_annot[c][1, :]
            if i == 0:
                events1 = events_tmp
            else:
                events1 = np.vstack([events1, events_tmp])
            i += 1
    # delete repeted triggers
    uniq, ind = np.unique(events1[:, 0], return_index=True)
    events = events1[ind, :]
    # Pick events closest to PD and replace them with PD_on time
    for pd_time in Ind_PD_ON:
        ind = np.argmin(np.absolute(events[:, 0] - pd_time))
        events[ind, 0] = pd_time
    event_id = {'Face/Face_only': 1, 'Face/Face_hands': 12,
                'Face/Face_shapes': 14, 'Face/Face_body': 15,
                'Hand/Hand_only': 2, 'Hand/Hand_face': 21,
                'Hand/Hand_body': 25, 'Hand/Hand_letters': 23,
                'Hand/Hand_shapes': 24, 'Body/Body_only': 5,
                'Body/Body_hands': 52, 'Body/Body_shapes': 54,
                'Body/Body_back': 55, 'Body/Body_distant': 255,
                'Other/letters': 3, 'Other/Lanscapes': 4,
                'Other/Landscape_letters': 43, 'Other/Landscape_hands': 42,
                'Other/Landscape_body': 45}
    
    return event_id, events
#########################################################
#####   Motion artifacts and head pos correction   ######
#########################################################


def _annotations_from_mask(times, art_mask, art_name):
    # make annotations - by Luke Bloy
    comps, num_comps = label(art_mask)
    onsets = []
    durations = []
    desc = []
    n_times = len(times)
    for l in range(1, num_comps+1):
        l_idx = np.nonzero(comps == l)[0]
        onsets.append(times[l_idx[0]])
        # duration is to the time after the last labeled time
        # or to the end of the times.
        if 1+l_idx[-1] < n_times:
            durations.append(times[1+l_idx[-1]] - times[l_idx[0]])
        else:
            durations.append(times[l_idx[-1]] - times[l_idx[0]])
        desc.append(art_name)
    return Annotations(onsets, durations, desc)


def annotate_motion(raw, pos, thr=0.01):
    """Find and annotate periods of high HPI distance w.r.t the median HPI pos
        and readjust trans matrix dev->head - written originally by Luke Bloy"""
    annot = Annotations([], [], [])

    info = raw.info
    time = pos[:, 0]
    quats = pos[:, 1:7]

    # Get static head pos from file, used to convert quat to cartesian
    chpi_locs_dev = sorted([d for d in info['hpi_results'][-1]
                            ['dig_points']], key=lambda x: x['ident'])
    chpi_locs_dev = np.array([d['r'] for d in chpi_locs_dev])
    # chpi_locs_dev[0]-> LPA, chpi_locs_dev[1]-> NASION, chpi_locs_dev[2]-> RPA
    # Get head pos changes during recording
    chpi_mov_head = np.array([_apply_quat(quat, chpi_locs_dev, move=True)
                              for quat in quats])

    # get median position across all recording
    chpi_mov_head_f = chpi_mov_head.reshape([-1, 9])  # always 9 chans
    chpi_med_head_tmp = np.median(chpi_mov_head_f, axis=0).reshape([3, 3])

    # get movement displacement from median
    hpi_disp = chpi_mov_head - np.tile(chpi_med_head_tmp, (len(time), 1, 1))
    # get positions above threshold distance
    disp = np.sqrt((hpi_disp ** 2).sum(axis=2))
    disp_exes = np.any(disp > thr, axis=1)

    # Get median head pos during recording under threshold distance
    weights = np.append(time[1:] - time[:-1], 0)
    weights[disp_exes] = 0
    weights /= sum(weights)
    tmp_med_head = weighted_median(chpi_mov_head, weights)
    # Get closest real pos to estimated median
    hpi_disp_th = chpi_mov_head - np.tile(tmp_med_head, (len(time), 1, 1))
    hpi_dist_th = np.sqrt((hpi_disp_th.reshape(-1, 9) ** 2).sum(axis=1))
    chpi_median_pos = chpi_mov_head[hpi_dist_th.argmin(), :, :]

    # Compute displacements from final median head pos
    hpi_disp = chpi_mov_head - np.tile(chpi_median_pos, (len(time), 1, 1))
    hpi_disp = np.sqrt((hpi_disp**2).sum(axis=-1))

    art_mask_mov = np.any(hpi_disp > thr, axis=-1)  # hpi_disp > thr why?
    annot += _annotations_from_mask(time, art_mask_mov,
                                    'Bad-motion-dist>%0.3f' % thr)

    # Compute new dev->head transformation from median
    init_dev_head_t = _quaternion_align(info['dev_head_t']['from'],
                                        info['dev_head_t']['to'],
                                        chpi_locs_dev, chpi_median_pos)
    dev_head_t = init_dev_head_t
    return annot, hpi_disp, dev_head_t


#########################################################
############      Generic functions       ###############
#########################################################


def weighted_median(data, weights):
    """ by tinybike
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights
    """
    dims = data.shape
    w_median = np.zeros((dims[1], dims[2]))
    for d1 in range(dims[1]):
        for d2 in range(dims[2]):
            data_dd = np.array(data[:, d1, d2]).squeeze()
            s_data, s_weights = map(np.array, zip(*sorted(zip(
                                                        data_dd, weights))))
            midpoint = 0.5 * sum(s_weights)
            if any(s_weights > midpoint):
                w_median[d1, d2] = (data[weights == np.max(weights)])[0]
            else:
                cs_weights = np.cumsum(s_weights)
                idx = np.where(cs_weights <= midpoint)[0][-1]
                if cs_weights[idx] == midpoint:
                    w_median[d1, d2] = np.mean(s_data[idx:idx+2])
                else:
                    w_median[d1, d2] = s_data[idx+1]
    return w_median


def csv_save(data, out_fname):
    with open(out_fname, "w") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(data)


def csv_read(out_fname):
    csv_dat = []
    with open(out_fname) as f:
        reader = csv.reader(f, delimiter=',')
        for col in reader:
            csv_dat.append(col)
    try:
        csv_dat = csv_dat[-1]
    except IndexError:
        csv_dat = csv_dat
    return csv_dat
