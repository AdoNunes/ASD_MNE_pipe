#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:02:59 2019

@author: nkozhemi
"""

import matplotlib.pyplot as plt
import os.path as op
import numpy as np
import sys
from mne.io import read_raw_ctf
from mne import pick_types
from mne.chpi import _calculate_head_pos_ctf, read_head_pos
sys.path.append('/Users/nkozhemi/Documents/Github/mne_chop_tools-master/mne_pipes')

from annotate_artifacts import (annotate_gfp_artifacts,
                                annotate_motion_artifacts,
                                plot_artifacts,
                                annotate_muscle_artifacts)

top_dir = '/Users/nkozhemi/Desktop/MEG_repo/MEG_children/18011010C/'
ds_fname = op.join(top_dir, 'MEG_Movie_20190319_09.ds') 

raw = read_raw_ctf(ds_fname, preload=True)
#raw.set_channel_types({'EEG057': 'eog', 'EEG058': 'ecg'})
#raw.info['bads'] = []

if raw.compensation_grade !=3:
    raw.apply_gradient_compensation(3)

# muscle artifacts
mus_annot, mus_raw = annotate_muscle_artifacts(raw, art_thresh=2.5,
                                               return_stat_raw=True)

# motion artifacts
pos = _calculate_head_pos_ctf(raw)
mov_annot, raw_hpi = annotate_motion_artifacts(raw, pos, return_stat_raw=True)

#pick meg channels and filter
picks = pick_types(raw.info, meg='mag', eeg=False, eog=False, stim=False)
raw.notch_filter(np.arange(60, 241, 60), picks=picks, fir_design='firwin')
raw.filter(1, 100, picks=picks)

# muscle artifacts
gfp_annot, stat_raw = annotate_gfp_artifacts(raw, return_stat_raw=True)


art_raw = stat_raw.copy();
art_raw.add_channels([raw_hpi, mus_raw]);
tresholds = {
"gfp_thresh" : 5,
"motion_disp_thresh" : 0.005,
"motion_velo_thresh" : 0.02,
"muscle_thresh" : 2
}

plot_artifacts(art_raw, tresholds)

#add annotation of muscle artifacts
#raw.set_annotations(mov_annot + mus_annot + gfp_annot)


#raw.plot(start=0, duration=10)
