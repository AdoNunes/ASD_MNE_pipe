

from os import path as op
from scipy.stats import zscore
import glob
import mne
import numpy as np
import csv
from annotate_artifacts import (annotate_motion_artifacts, plot_artifacts)
from mne.annotations import Annotations, read_annotations
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage.measurements import label

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
    9- source



    out directory:
        /bad_chn_annot
        /ICA

    """

    def __init__(self, subject, experiment, paths_dic):
        self.subject = subject
        self.experiment = experiment
        self.pth_root = op.expanduser(paths_dic["root"])
        self.pth_out = op.expanduser(paths_dic["out"])
        self.check_outdir()
        self.pth_subject = op.join(self.pth_root, subject)
        self.pth_raw = glob.glob(op.join(self.pth_subject, subject) + '_' +
                                 experiment + '*')[-1]
        self.raw = mne.io.read_raw_ctf(self.pth_raw, preload=False)
        if self.raw.compensation_grade != 3:
            self.raw.apply_gradient_compensation(0)

    def check_outdir(self):
        from os import makedirs
        out_dir = self.pth_out
        self.out_bd_ch = op.join(out_dir, 'bad_chans')
        self.out_annot = op.join(out_dir, 'annots')
        self.out_ICAs = op.join(out_dir, 'ICAs')
        makedirs(self.out_bd_ch, exist_ok=True)  # I want to loop it
        makedirs(self.out_annot, exist_ok=True)
        makedirs(self.out_ICAs, exist_ok=True)

# TODO save annotations and just load them if exist
    def detectMov(self, threshold_mov=.005, do_plot=True, overwrite=False):
        fname = self.subject + '_' + self.experiment + '_mov.csv'
        out_csv_f = op.join(self.out_bd_ch, fname)
        if op.exists(out_csv_f) and not overwrite:
            mov_annot = read_annotations(out_csv_f)
            print('Reading from file, mov segments are:', mov_annot)
        else:
            pos = mne.chpi._calculate_head_pos_ctf(self.raw)
            if do_plot is True:
                mov_annot, araw = annotate_motion_artifacts(self.raw, pos,
                                  disp_thr=threshold_mov, velo_thr=None,
                                  gof_thr=None, return_stat_raw=True)
                tresholds = {'motion_disp_thresh': threshold_mov}
                plot_artifacts(araw, tresholds)
            else:
                araw = annotate_motion_artifacts(self.raw, pos,
                                                 disp_thr=threshold_mov,
                                                 velo_thr=None, gof_thr=None)
            mov_annot.save(out_csv_f)
        self.raw.set_annotations(mov_annot)

    def detectBadChannels(self, zscore_v=4, save_csv=None, overwrite=False):
        """ zscore_v = zscore threshold, save_csv: path_tosaveCSV
        """
        fname = self.subject + '_' + self.experiment + '_bads.csv'
        out_csv_f = op.join(self.out_bd_ch, fname)
        if op.exists(out_csv_f) and not overwrite:
            bad_chns = self.csv_read(out_csv_f)
            print('Reading from file, bad chans are:', bad_chns)
            self.raw.info['bads'] = bad_chns
        else:
            from itertools import compress
            raw_copy = self.raw.copy().crop(30., 220.).load_data()
            raw_copy = raw_copy.pick_types(meg=True, ref_meg=False)\
                .filter(1, 45).resample(150, npad='auto')
            max_Pow = np.sqrt(np.sum(raw_copy.get_data() ** 2, axis=1))
            max_Z = zscore(max_Pow)
            max_th = max_Z > zscore_v
            bad_chns = list(compress(raw_copy.info['ch_names'], max_th))
            if bad_chns:
                raw_copy.plot(n_channels=100, block=True)
                bad_chns = raw_copy.info['bads']
            if save_csv is not None:
                self.csv_save(bad_chns, out_csv_f)
            self.raw.info['bads'] = bad_chns
            self.ch_max_Z = max_Z

    def csv_save(self, data, out_fname):
        with open(out_fname, "w") as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(data)

    def csv_read(self, out_fname):
        csv_dat = []
        with open(out_fname) as f:
            reader = csv.reader(f, delimiter=',')
            for col in reader:
                csv_dat.append(col)
        return csv_dat[-1]

    def run_ICA(self, overwrite=False):
        fname = self.subject + '_' + self.experiment + '-ica.fif.gz'
        out_fname = op.join(self.out_ICAs, fname)
        if op.exists(out_fname) and not overwrite:
            self.ica = mne.preprocessing.read_ica(out_fname)
        else:
            from mne.preprocessing import ICA
            raw_copy = self.raw.copy().load_data().filter(1, 45)
            self.ica = ICA(method='fastica', random_state=42,
                           n_components=0.99, max_iter=500)
            picks = mne.pick_types(raw_copy.info, meg=True, ref_meg=False,
                                   stim=False, exclude='bads')
            reject = dict(grad=4000e-13, mag=4e-12)  # what rejec intervals?
            self.ica.fit(raw_copy, picks=picks, reject=reject, decim=3)

    def plot_ICA(self):
        raw_copy = self.raw.copy().load_data().filter(1, 45) \
                    .pick_types(meg=True, ref_meg=False)
        self.ica.plot_components(picks=np.arange(19))
        self.ica.plot_sources(raw_copy, block=True)

    def save_ICA(self, overwrite=False):
        fname = self.subject + '_' + self.experiment + '-ica.fif.gz'
        out_fname = op.join(self.out_ICAs, fname)
        if not op.exists(out_fname) or overwrite:
            self.ica.save(out_fname)

    def get_events(self, plot=1):
        # general description of the data
        raw_copy = self.raw.copy()
        task = self.experiment
        fs = raw_copy.info['sfreq']
        time = raw_copy.buffer_size_sec
        N_samples = raw_copy.n_times
        print('Data is composed from')
        print(str(N_samples) + ' samples with ' + str(fs) + ' Hz sampl rate')
        print('Total time = ' + str(time) + 's')
        # get photodiode events from CTF data
        PD_ts, Ind_PD_ON, Ind_PD_OFF, T_PD = get_photodiode_events(raw_copy)
        # pick Trigger channel time series from CTF data
        Trig = mne.io.pick.pick_channels_regexp(raw_copy.info['ch_names'],
                                                'UPPT001')
        Trig_ts = raw_copy.get_data(picks=Trig)
        # get events from trigger channel
        events_trig = mne.find_events(raw_copy, stim_channel='UPPT001')

        if task == 'Car':
            event_id = {'Transp/H2L': 10, 'Transp/L2H': 20,
                        'NotTransp/H2L': 30, 'NotTransp/L2H': 40}
            # get trigger names for PD ON states
            events = get_triger_names_PD(event_id, Ind_PD_ON, events_trig)

        elif task == 'Movie':
            if len(Ind_PD_ON) and len(Ind_PD_OFF) != 196:
                print('NOT ALL OF THE PD STIM PRESENT!!!')
            event_id = []
            events = np.zeros((len(Ind_PD_ON), 3))
            events[:, 0] = Ind_PD_ON
            events[:, 2] = 1

        elif task == 'Flanker':
            event_id = {'Incongruent/Left': 3, 'Incongruent/Right': 5,
                        'Congruent/Left': 4, 'Congruent/Right': 6,
                        'Reward/Face': 7, 'Reward/Coin': 8,
                        'Reward/Neut_FaceTRL': 9, 'Reward/Neut_CoinTRL': 10}
            # get trigger names for PD ON states
            events = get_triger_names_PD(event_id, Ind_PD_ON, events_trig)

        events = events.astype(np.int64)
        # plot trig and PD
        if plot:
            plt.figure()
            plot_events(PD_ts, Ind_PD_ON, T_PD, Ind_PD_OFF, Trig_ts, events,
                        task)
        return event_id, events

    def detect_muscartif(self, art_thresh=2, t_min=2,
                                  desc='Bad-muscle', n_jobs=1,
                                  return_stat_raw=False, plot=True):
        """Find and annotation mucsle artifacts."""
        raw = self.raw.copy().load_data()
        # pick meg_chans
        raw.info['comps'] = []
        raw.pick_types(meg=True, ref_meg=False)
        raw.filter(110, 140, n_jobs=n_jobs, fir_design='firwin')
        raw.apply_hilbert(n_jobs=n_jobs, envelope=True)
        sfreq = raw.info['sfreq']
        art_scores = stats.zscore(raw._data, axis=1)
        stat_raw = None
        art_scores_filt = mne.filter.filter_data(art_scores.mean(axis=0),
                                                 sfreq, None, 5)
        art_mask = art_scores_filt > art_thresh
        if return_stat_raw:
            tmp_info = mne.create_info(['mucsl_score'], raw.info['sfreq'],
                                       ['misc'])
            stat_raw = mne.io.RawArray(art_scores_filt.reshape(1, -1),
                                       tmp_info)

        # remove artifact free periods under limit
        idx_min = t_min * sfreq
        comps, num_comps = label(art_mask == 0)
        for l in range(1, num_comps+1):
            l_idx = np.nonzero(comps == l)[0]
            if len(l_idx) < idx_min:
                art_mask[l_idx] = True  
        if plot:
            raw = self.raw.copy().load_data()
            raw.set_annotations(_annotations_from_mask(raw.times, art_mask,
                                                       desc))
            raw.plot()
        return _annotations_from_mask(raw.times, art_mask, desc), stat_raw

##################################################################
# Photod Diode functions
##################################################################

def get_photodiode_events(raw):

    raw = raw.copy()
    # pick PD channel time series from CTF data
    raw.info['comps'] = []
    PD = mne.io.pick.pick_channels_regexp(raw.info['ch_names'], 'UADC015-3007')
    PD_ts = raw.get_data(picks=PD)

    # find all 'ON' states of PD
    n, bins, patches = plt.hist(PD_ts[0, :], bins=50,
                                range=(np.percentile(PD_ts[0, :], 1),
                                np.percentile(PD_ts[0, :], 99)),
                                color='#0504aa', alpha=0.7, rwidth=0.85)
    thr = bins[np.nonzero(n == np.min(n))]  # set a treshold
    if len(thr) > 1:
        thr = thr[-1]
    T_PD = PD_ts > thr
    Ind_PD_ON = []
    Ind_PD_OFF = []
    for ind, n in enumerate(T_PD[0, :]):
        if (n == True and T_PD[0, ind-1] == False and np.all(T_PD[0, ind-10:ind-1] == False)):
            Ind_PD_ON.append(ind)
        elif (n == False and T_PD[0, ind-1] == True and np.all(T_PD[0, ind-10:ind-1] == True)):
            Ind_PD_OFF.append(ind)
    return PD_ts, Ind_PD_ON, Ind_PD_OFF, T_PD


def plot_events(PD_ts, Ind_PD_ON, T_PD, Ind_PD_OFF, Trig_ts, events, task):

    plt.plot((PD_ts[0, :]))
    plt.plot(T_PD[0, :]*5)
    y = np.ones((1, len(Ind_PD_ON)))
    y = y = 0.5*y
    plt.plot(np.array(Ind_PD_ON), y[0, :], 'o', color='black')
    y = np.ones((1, len(Ind_PD_OFF)))
    y = y = 0.5*y
    plt.plot(np.array(Ind_PD_OFF), y[0, :], 'o', color='red')
    plt.plot(events[:, 0], events[:, 2], 'o', color='green')
    plt.plot((Trig_ts[0, :]))
    plt.ylabel('a.u.')
    plt.xlabel('samples')
    plt.title('Photodiode and triggers timing for ' + task + ' task')


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
            m = min(inx[inx > 0])
            events[inx == m, 2] = value
    return events


def _annotations_from_mask(times, art_mask, art_name):
    # make annototations
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
