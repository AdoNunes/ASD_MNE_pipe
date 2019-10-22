

from os import path as op
from scipy.stats import zscore
import glob
import mne
import numpy as np
import csv
import threading



class MNEprepro():

    
    """

    Class to preproces CTF data

    Usage:

    raw_prepro = MNEprepro(subject, experiment, paths_dic)
    
    paths_dic = {
        "root": "/Volumes/Data_projec/data/REPO/MEG_repo",
        "root": "~/Desktop/projects/MNE/data",
        "meg": "MEG",
        "subj_anat": 'anatomy'
        "out": "~/Desktop/projects/MNE/data_prep"
    }
    
    subject = '18011014C'
    experiment = 'Movie'
        
    
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

    def detectBadChannels(self, zscore_v=3, save_csv=None, overwrite=False):
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
                .filter(1, 50).resample(150, npad='auto')
            max_Pow = np.sqrt(np.sum(raw_copy.get_data() ** 2, axis=1))
            max_Z = zscore(max_Pow)
            max_th = max_Z > zscore_v
            bad_chns = list(compress(raw_copy.info['ch_names'], max_th))
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

    def detectMov(self):
        thr = .5  # mm
        pos = mne.chpi._calculate_head_pos_ctf(self.raw)

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
