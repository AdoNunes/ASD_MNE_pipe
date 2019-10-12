

from os import path as op
import glob
import mne

class MNEprepro():

    """

    Class to preproces CTF data

    Usage:

    raw_prepro = MNE_prepro(subject, experiment, paths_dic)
    paths_dic = {
        "root": "/Volumes/Data_projec/data/REPO/MEG_repo",
        "root": "~/Desktop/projects/MNE/data",
        "meg": "MEG",
        "subj_anat": 'anatomy'
    }
    
    subject = '18011014C'
    experiment = 'Movie'
        
    """

    def __init__(self,subject,experiment,paths_dic):
        self.subject = subject
        self.experiment = experiment
        self.pth_root = op.expanduser(paths_dic["root"])
        self.pth_subject = op.join(self.pth_root, subject)
        self.pth_raw = glob.glob(op.join(self.pth_subject , subject)+'_'+
        experiment +'*')
        self.raw = mne.io.read_raw_ctf(self.pth_raw, preload=True)

    def data_plot_meg(self):
        self

    def data_filter(self, f_low=None, f_high=None, method='iir'):
        print ("Filtering data from ", f_low,"to", f_high)
        self.raw.filter(1., None, fir_design='firwin')
        