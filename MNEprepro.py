

from os import path as op
from scipy.stats import zscore
import glob
import mne
import numpy as np

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

    def check_outdir(self):
        from os import makedirs
        out_dir = self.pth_out
        self.out_bd_ch = op.join(out_dir, 'bad_chans')
        self.out_annot = op.join(out_dir, 'annots')
        self.out_ICAs = op.join(out_dir, 'ICAs')
        makedirs(self.out_bd_ch, exist_ok=True)
        makedirs(self.out_annot, exist_ok=True)
        makedirs(self.out_ICAs, exist_ok=True)

    def detectBadChannels(self, zscore_v=3):
        from itertools import compress
        raw_copy = self.raw.copy().pick_types(meg=True, ref_meg=False)
        raw_copy = raw_copy.crop(30., 220.).load_data().filter(1, 50) \
            .resample(150, npad='auto')
        max_Pow = np.sqrt(np.sum(raw_copy.get_data() ** 2, axis=1))
        max_Z = zscore(max_Pow)
        max_th = int(max_Z > zscore_v)
        bad_chns = list(compress(raw_copy.info['ch_names'], max_th))
        return bad_chns, max_Z

    def save_bad_chan(self, to_ds=False, to_deriv=True):
        self

    def data_filter(self, f_low=None, f_high=None, method='iir'):
        print("Filtering data from ", f_low, "to", f_high)
        self.raw.filter(1., None, fir_design='firwin')
