

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
import sys


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
        self.check_outdir()
        self.pth_subject = op.join(self.pth_root, subject)
        self.pth_raw = glob.glob(op.join(self.pth_subject, subject) + '_' +
                                 experiment + '*')[-1]
        self.raw = mne.io.read_raw_ctf(self.pth_raw, preload=False)
        if self.raw.compensation_grade != 3:
            self.raw.apply_gradient_compensation(3)

    def check_outdir(self):
        from os import makedirs
        out_dir = self.pth_out
        self.out_bd_ch = op.join(out_dir, 'bad_chans')
        self.out_annot = op.join(out_dir, 'annots')
        self.out_ICAs = op.join(out_dir, 'ICAs')
        makedirs(self.out_bd_ch, exist_ok=True)  # I want to loop it
        makedirs(self.out_annot, exist_ok=True)
        makedirs(self.out_ICAs, exist_ok=True)

    def detectBadChannels(self, zscore_v=4, save_csv=True, overwrite=False):
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
            print('Looking for bad channels')
            raw_copy = self.raw.copy().crop(30., 220.).load_data()
            raw_copy = raw_copy.pick_types(meg=True, ref_meg=False)\
                .filter(1, 45).resample(150, npad='auto')
            max_Pow = np.sqrt(np.sum(raw_copy.get_data() ** 2, axis=1))
            max_Z = zscore(max_Pow)
            max_th = max_Z > zscore_v
            bad_chns = list(compress(raw_copy.info['ch_names'], max_th))
            raw_copy.info['bads'] = bad_chns
            if bad_chns:
                print('Plotting data,bad chans are:', bad_chns)
                raw_copy.plot(n_channels=100, block=True, bad_color='r')
                bad_chns = raw_copy.info['bads']
                print('Bad chans are:', bad_chns)
            else:
                print('No bad chans found')
            if save_csv is True:
                self.csv_save(bad_chns, out_csv_f)
            self.raw.info['bads'] = bad_chns
            self.ch_max_Z = max_Z

    def detectMov(self, thr_mov=.005, do_plot=True, overwrite=False):
        from mne.transforms import read_trans
        fname = self.subject + '_' + self.experiment + '_mov.csv'
        out_csv_f = op.join(self.out_annot, fname)
        fname_t = self.subject + '_' + self.experiment + '_dev2head-trans.fif'
        out_csv_f_t = op.join(self.out_annot, fname_t)
        if op.exists(out_csv_f) and not overwrite:
            mov_annot = read_annotations(out_csv_f)
            print('Reading from file, mov segments are:', mov_annot)
            print('Reading from file, dev to head transformation')
            dev_head_t = read_trans(out_csv_f_t)
        else:
            pos = mne.chpi._calculate_head_pos_ctf(self.raw)
            mov_annot, hpi_disp, dev_head_t = annotate_motion(self.raw, pos,
                                                              thr=thr_mov)
            if do_plot is True:
                plt.figure()
                plt.plot(hpi_disp)
                plt.show()
            mov_annot.save(out_csv_f)
            dev_head_t.save(out_csv_f_t)
        self.raw.set_annotations(mov_annot)
        self.raw.info['dev_head_t_old'] = self.raw.info['dev_head_t']
        self.raw.info['dev_head_t'] = dev_head_t

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
            event_id = {'SceneOnset': 1}
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

    def epoching(self, event_id, events, tmin=-0.4, tmax=0.5):
        raw_copy = self.raw.copy().load_data().filter(1, 45) \
                    .pick_types(meg=True, ref_meg=False)
        epochs = mne.Epochs(raw_copy, events=events, event_id=event_id,
                            tmin=tmin, tmax=tmax, baseline=(tmin, 0.0),
                            picks=('meg'))
        return epochs

    def detect_muscartif(self, art_thresh=2, t_min=2,
                         desc='Bad-muscle', n_jobs=1, return_stat_raw=False,
                         plot=True):
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

        # remove artifact free periods under threshold
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


#########################################################
########## Motion artifacts and head pos correction
#########################################################

from mne.io import RawArray
from mne.io.ctf.trans import _quaternion_align

from mne.annotations import Annotations
from mne.chpi import _apply_quat
from mne.transforms import apply_trans


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


def annotate_motion(raw, pos, thr=0.01):
    """Find and annotate periods of high HPI velocity and high HPI distance.
        written originally by Luke Bloy"""
    annot = Annotations([], [], [])

    info = raw.info
    time = pos[:, 0]
    quats = pos[:, 1:7]

    # Get static head pos from file
    chpi_locs_dev = sorted([d for d in info['hpi_results'][-1]
                            ['dig_points']], key=lambda x: x['ident'])
    chpi_locs_dev = np.array([d['r'] for d in chpi_locs_dev])
    # chpi_locs_dev[0]-> LPA, chpi_locs_dev[1]-> NASION, chpi_locs_dev[2]-> RPA
    chpi_static_head = apply_trans(info['dev_head_t'], chpi_locs_dev)
    # Get head pos changes during recording
    chpi_mov_head = np.array([_apply_quat(quat, chpi_locs_dev, move=True)
                              for quat in quats])
    # Remove static head to get rel. movement
    hpi_disp = chpi_mov_head - np.tile(chpi_static_head, (len(time), 1, 1))
    # get positions where movement below threshold
    mov_exes = np.any(np.any(np.absolute(hpi_disp) > thr, axis=2), axis=1)

    # Get median head pos during recording excluding excessive mov periods
    weights = np.append(time[1:] - time[:-1], 0)
    weights[mov_exes] = 0
    weights /= sum(weights)
    tmp_med_head = weighted_median(chpi_mov_head, weights)
    # Get closest pos to median
    hpi_disp_th = chpi_mov_head - np.tile(tmp_med_head, (len(time), 1, 1))
    hpi_dist_th = np.sqrt((hpi_disp_th.reshape(-1, 9) ** 2).sum(axis=1))
    chpi_median_pos = chpi_mov_head[hpi_dist_th.argmin(), :, :]

    # Compute displacements
    hpi_disp = chpi_mov_head - np.tile(chpi_median_pos, (len(time), 1, 1))
    hpi_disp = np.sqrt((hpi_disp**2).sum(axis=-1))

    art_mask_mov = hpi_disp > thr
    annot += _annotations_from_mask(time, art_mask_mov,
                                    'Bad-motion-dist>%0.3f' % thr)
    # Compute new dev->head transformation
    init_dev_head_t = _quaternion_align(info['dev_head_t']['from'],
                                        info['dev_head_t']['to'],
                                        chpi_locs_dev, chpi_median_pos)
    dev_head_t = init_dev_head_t
    return annot, hpi_disp, dev_head_t


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


def annotate_motion_old(raw, pos, disp_thr=0.01,gof_thr=0.99,
                    return_stat_raw=False):
    """Find and annotate periods of high HPI velocity and high HPI distance.
        written by Luke Bloy"""
    annot = Annotations([], [], [])

    info = raw.info
    # grab initial cHPI locations
    # point sorted in hpi_results are in mne device coords
    chpi_locs_dev = sorted([d for d in info['hpi_results'][-1]
                            ['dig_points']], key=lambda x: x['ident'])
    chpi_locs_dev = np.array([d['r'] for d in chpi_locs_dev])
    # chpi_locs_dev[0] -> LPA
    # chpi_locs_dev[1] -> NASION
    # chpi_locs_dev[2] -> RPA
    chpi_static_head = apply_trans(info['dev_head_t'], chpi_locs_dev)

    time = pos[:, 0]
    n_hpi = chpi_static_head.shape[0]
    quats = pos[:, 1:7]
    chpi_moving_head = np.array([_apply_quat(quat, chpi_locs_dev, move=True)
                                 for quat in quats])
    # Get median head pos during recording
    tmp_med_head = np.median(chpi_moving_head, axis=0)
    hpi_disp = chpi_moving_head - np.tile(tmp_med_head, (len(time), 1, 1))
    hpi_disp_dist = (hpi_disp.reshape(-1, hpi_disp.shape[1]*hpi_disp.shape[2]) 
                  ** 2).sum(axis=1)
    chpi_median_pos= chpi_moving_head[hpi_disp_dist.argmin(),:,:]

    # compute displacements
    hpi_disp = chpi_moving_head - np.tile(chpi_median_pos, (len(time), 1, 1))
    hpi_disp = np.sqrt((hpi_disp**2).sum(axis=-1))

    art_mask = hpi_disp > disp_thr
    annot += _annotations_from_mask(time, art_mask,
                                    'Bad-motion-dist>%0.3f' % disp_thr)

    art_mask = pos[:, 7] <= gof_thr
    annot += _annotations_from_mask(time, art_mask,
                                    'Bad-chpi_gof>%0.3f' % gof_thr)

    tmp = 1000 * hpi_disp.max(axis=0)
    _fmt = '\tHPI00 - %0.1f'
    for i in range(1, n_hpi):
        _fmt += '\n\tHPI%02d' % (i) +' - %0.1f'
    logger.info('CHPI MAX Displacments (mm):')
    logger.info(_fmt % tuple(tmp))

    raw_hpi = None
    if return_stat_raw:
        n_times = len(raw.times)
        # build full time data arrays
        start_idx = raw.time_as_index(time, use_rounding=True)
        end_idx = raw.time_as_index(np.append(time[1:], raw.times[-1]),
                                    use_rounding=True)
        data_pos = np.zeros((n_hpi, n_times))
        for t_0, t_1, disp_val, velo_val in zip(start_idx, end_idx,
                                                hpi_disp):
            t_slice = slice(t_0, t_1)
            data_pos[:n_hpi, t_slice] = np.tile(disp_val, (t_1 - t_0, 1)).T

        ch_names = []
        for i in range(n_hpi):
            ch_names.append('HPI%02d_disp_pos' % i)

        # build raw object!
        info = create_info(
                        ch_names=ch_names,
                        ch_types=np.repeat('misc', len(ch_names)),
                        sfreq=raw.info['sfreq'])
        raw_hpi = RawArray(data_pos, info)

    return annot, raw_hpi