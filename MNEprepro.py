
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

    def detect_movement(self, thr_mov=.01, plot=True, overwrite=False):
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
            mov_annot.save(out_csv_f)
            dev_head_t.save(out_csv_f_t)
            fig.savefig(out_csv_f[:-4]+'.png')
        self.raw.set_annotations(mov_annot)
        self.raw.info['dev_head_t_old'] = self.raw.info['dev_head_t']
        self.raw.info['dev_head_t'] = dev_head_t
        self.annot_movement = mov_annot

    def detect_muscle(self, thr=1.5, t_min=2, plot=True, overwrite=False):
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
                raw.set_annotations(raw.annotations + mus_annot)
                raw.plot(n_channels=140, block=True, bad_color='r', exclude=[])
                mus_annot = raw.annotations
                if not (old_bd_chns == raw.info['bads']):
                    bad_chns = raw.info['bads']
                    print('Saving new bad channels list \n ')
                    print('Bad chans are:', bad_chns)
                    fname = self.subject + '_' + self.experiment + '_bads.csv'
                    csv_save(bad_chns, op.join(self.out_bd_ch, fname))
            mus_annot.save(out_csv_f)
        self.raw.set_annotations(self.raw.annotations + mus_annot)
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
            reject = dict(grad=4000e-13, mag=4e-12)  # what rejec intervals?
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
            self.ica.done = False
        else:
            self.run_ICA(self)
        # Check if ICA comps were inspected
        data_not_clean = True
        if check_if_done is True:
            if self.ica.done is True:
                data_not_clean = False
        # Plot interactively to select bad comps
        if data_not_clean is True:
            raw_copy = self.raw.copy().load_data().filter(1, 45)
        while data_not_clean is True:
            # ICA comp plotting
            self.ica.plot_components(inst=raw_copy)
            self.ica.plot_sources(raw_copy, block=True)
            # Clean and raw sensor plotting
            raw_copy.plot(n_channels=136, title='NO ICA')
            raw_ica = raw_copy.copy().pick_types(meg=True, ref_meg=False)
            self.ica.apply(raw_ica)
            raw_ica.plot(n_channels=136, title='ICA cleaned', block=True)
            data_not_clean = bool(int(input("Select other ICA components? "
                                            "[0-no, 1-yes]: ")))

    def get_events(self, plot=True):
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
        PD_ts, Ind_PD_ON, Ind_PD_OFF, T_PD = get_photodiode_events(raw_copy,fs)
        # pick Trigger channel time series from CTF data
        Trig = mne.io.pick.pick_channels_regexp(raw_copy.info['ch_names'],
                                                'UPPT001')
        Trig_ts = raw_copy.get_data(picks=Trig)
        # get events from trigger channel
        events_trig = mne.find_events(raw_copy, stim_channel='UPPT001',
                                      shortest_event=1)
        
        print(str(len(Ind_PD_ON)) + ' PD ONSETS FOUND')
        
                
        if task == 'Car':
            event_id = {'Transp/H2L': 10, 'Transp/L2H': 20,
                        'NotTransp/H2L': 30, 'NotTransp/L2H': 40}
            # get trigger names for PD ON states
            events = get_triger_names_PD(event_id, Ind_PD_ON, events_trig)
            
        elif task == 'Movie':
            
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
                        task, ID)
        return event_id, events

    def epoching(self, event_id, events, tmin=-0.4, tmax=0.5):
        raw_copy = self.raw.copy().load_data().filter(1, 45) \
                    .pick_types(meg=True, ref_meg=False)
        epochs = mne.Epochs(raw_copy, events=events, event_id=event_id,
                            tmin=tmin, tmax=tmax, baseline=(tmin, 0.0),
                            picks=('meg'))
        return epochs


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
              np.all(T_PD[0, ind+min_samp8:ind+min_samp4] == False)):
            Ind_PD_OFF.append(ind)
    return PD_ts, Ind_PD_ON, Ind_PD_OFF, T_PD


def plot_events(PD_ts, Ind_PD_ON, T_PD, Ind_PD_OFF, Trig_ts, events, task, ID):

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
