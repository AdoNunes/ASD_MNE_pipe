#!/bin/bash

# run this script specifying fresurfer home directory and meg data directory
# note, meg data directory should have anatomy dir with T1 image

SUBJECTS_DIR=/Users/nkozhemi/Desktop/REPO/Freesurfer/
MEG_DIR=/Users/nkozhemi/Desktop/REPO/MEG_repo/MEG/
for subjectid in $(ls $MEG_DIR); do

if [ -d $MEG_DIR/$subjectid/anatomy ]; then
if ! [ -d $SUBJECTS_DIR/$subjectid ]; then

mri=$MEG_DIR/$subjectid/anatomy/$subjectid\_MEG_anatomy_anatomical.mgz

recon-all -all -subjid $subjectid -i $mri

mne make_scalp_surfaces --overwrite --subject $subjectid 
mne watershed_bem --overwrite --subject $subjectid 

fi
fi
done