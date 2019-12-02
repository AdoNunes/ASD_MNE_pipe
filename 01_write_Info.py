#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 07:51:19 2019

@author: nkozhemi

This script save subjects info in fif format in Fresurfer folder. Info file is
subsequently used during corregistration

Note! before running this script make sure to run MNE_anatomical.sh from your
terminal to have the BEMs and surfaces ready for each subject

All you have to do:
    1 - Run MNE_anatomical.sh from terminal
    2 - Run this script
    3 - run mne coreg from the terminal
"""

import appnope
from MNEprepro import MNEprepro
import socket
import glob
import os.path as op
import os
from shutil import copyfile
import mne
appnope.nope()


# %% Create variables for class object
paths_dic = {  # "root": "/Volumes/Data_projec/data/REPO/MEG_repo",
            "root": "~/Desktop/projects/MNE/data",
            "meg": "MEG",
            "subj_anat": 'anatomy',
            "out": "~/Desktop/projects/MNE/data_prep",
            "FS": "~/Desktop/REPO/Fresurfer"
            }

Host = (socket.gethostname())

if Host == 'owners-MacBook-Pro.local':
    paths_dic['root'] = "/Volumes/4TB_drive/projects/MEG_repo/MEG_children_rs"
    paths_dic['out'] = "~/Desktop/projects/MNE/data_prep"
elif Host == 'sc-155028' or 'sc-155014':
    paths_dic['root'] = "~/Desktop/REPO/MEG_repo/MEG_children_rs"  # MEG folder
    paths_dic['out'] = "~/Desktop/projects/MNE/data_prep"  # output folder
    paths_dic['FS'] = "~/Desktop/REPO/MEG_repo/Freesurfer_children"  # Freesurfer output folder

# get the Subject list from Freeserfer folder
pth_tmp = op.join(op.expanduser(paths_dic["FS"]), '18*')
Subj_list = glob.glob(pth_tmp)

# Specify the task -  'Movie', 'CarTask', 'Flanker'
experiment = 'CarTask'

# %% run the loop for each subject
for iSubj in Subj_list:

    subject = op.basename(iSubj)
    print('Preprocessing subject: ' + subject)

# Create Class object to get the paths to .pos file and .ds
    try:
        raw_prepro = MNEprepro(subject, experiment, paths_dic)
    except IndexError:
        continue

# temporarily copy pos file to .ds folder
    th_tmp = op.join(raw_prepro.pth_subject, '*.pos')
    pos_src = ''.join(glob.glob(th_tmp))
    pos_dest = op.join(raw_prepro.pth_raw + '/' + op.basename(''.join(pos_src)))
    copyfile(pos_src, pos_dest)

# Create Class object with updated INFO (now it has digitalization points)
    raw_prepro = MNEprepro(subject, experiment, paths_dic)

# Write info file inside Freesurfer output folder of current subject
    info = raw_prepro.raw.info
    info_fname = iSubj + '/' + subject + '_' + experiment + '-info.fif'
    mne.io.write_info(info_fname, raw_prepro.raw.info)

# remove tmp .pos file from .ds folder
    os.remove(pos_dest)
