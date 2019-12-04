
''' Create dic with pipeline paths'''


def set_paths():
    import socket
    paths_dic = {  # "root": "/Volumes/Data_projec/data/REPO/MEG_repo",
            "root": "~/Desktop/projects/MNE/data",
            "meg": "MEG",
            "subj_anat": 'anatomy',
            "out": "~/Desktop/projects/MNE/data_prep"
        }

    # Based on the computer name set specific paths
    Host = (socket.gethostname())

    if Host == 'owners-MacBook-Pro.local':
        paths_dic['root'] = "/Volumes/4TB_drive/projects/MEG_repo/" \
                            "MEG_children_rs"
        paths_dic['out'] = "~/Desktop/projects/MNE/data_prep"
        import appnope
        appnope.nope()
    elif Host == 'sc-157028' or Host == 'sc-155014':
        paths_dic['root'] = "~/Desktop/REPO/MEG_repo/MEG_children_rs"
        paths_dic['FS'] = "~/Desktop/REPO/MEG_repo/Freesurfer_children"
        paths_dic['out'] = "~/Desktop/projects/MNE/data_prep"
        import appnope
        appnope.nope()
    elif Host == 'megryan.nmr.mgh.harvard.edu':
        path_gen = '/local_mount/space/megryan/2/users/adonay/projects/ASD/'
        paths_dic['root'] = path_gen + "/MEG_children_rs"
        paths_dic['out'] = path_gen + "/data_prep"
        paths_dic['FS'] = path_gen + "/Freesurfer_children"

    paths_dic['data2src'] = paths_dic['out'] + "/data2src"

    return paths_dic

