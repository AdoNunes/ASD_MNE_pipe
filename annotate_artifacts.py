
import logging
from functools import partial
import matplotlib.pyplot as plt
import os.path as op

from mne import create_info, pick_channels_regexp, pick_channels
from mne.io import RawArray
from mne.io.ctf.res4 import _read_res4
from mne.io.ctf.trans import _make_ctf_coord_trans_set, _quaternion_align
from mne.utils import logger
from mne.annotations import Annotations
from mne.chpi import (_calculate_head_pos_ctf, _apply_quat,
                      _unit_quat_constraint, _get_hpi_initial_fit)

from mne.transforms import (apply_trans, rot_to_quat, quat_to_rot, Transform,
                            invert_transform)

from scipy import stats
import numpy as np
from mne.filter import filter_data
from scipy.ndimage.measurements import label


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


def annotate_muscle_artifacts(raw, art_thresh=0.075, t_min=2,
                              desc='Bad-muscle', n_jobs=1,
                              return_stat_raw=False):
    """Find and annotation mucsle artifacts."""
    raw = raw.copy()
    # pick meg_chans
    raw.info['comps'] = []
    raw.pick_types(meg=True, ref_meg=False)
    raw.filter(110, 140, n_jobs=n_jobs, fir_design='firwin')
    raw.apply_hilbert(n_jobs=n_jobs, envelope=True)
    sfreq = raw.info['sfreq']
    art_scores = stats.zscore(raw._data, axis=1)
    stat_raw = None
    art_scores_filt = filter_data(art_scores.mean(axis=0), sfreq, None, 5)
    art_mask = art_scores_filt > art_thresh
    if return_stat_raw:
        tmp_info = create_info(['mucsl_score'], raw.info['sfreq'], ['misc'])
        stat_raw = RawArray(art_scores_filt.reshape(1, -1), tmp_info)

    # remove artifact free periods under limit
    idx_min = t_min * sfreq
    comps, num_comps = label(art_mask == 0)
    for l in range(1, num_comps+1):
        l_idx = np.nonzero(comps == l)[0]
        if len(l_idx) < idx_min:
            art_mask[l_idx] = True
    return _annotations_from_mask(raw.times, art_mask, desc), stat_raw


def annotate_gfp_artifacts(raw, art_thresh=2.5,
                              desc='Bad-GFP', n_jobs=1,
                              return_stat_raw=False):

    raw_gfp = np.sqrt((raw.copy().pick_types(meg=True, ref_meg=False).get_data()**2).sum(axis=0))
    raw_gfp /= np.median(raw_gfp)

    art_mask = raw_gfp > art_thresh
    stat_raw = None
    if return_stat_raw:
        tmp_info = create_info(['gfp_normed'], raw.info['sfreq'], ['misc'])
        stat_raw = RawArray(raw_gfp.reshape(1, -1), tmp_info)
        # stat_raw = RawArray(raw_gfp.reshape(1, -1), raw.info)
        # stat_raw.add_channels([tmp_raw], force_update_info=True)
        # stat_raw = tmp_raw
    return _annotations_from_mask(raw.times, art_mask, desc), stat_raw


def annotate_motion_artifacts(raw, pos, disp_thr=0.01, velo_thr=0.03,
                              gof_thr=0.99, return_stat_raw=False):
    """Find and annotate periods of high HPI velocity and high HPI distance."""
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

    # compute displacements
    hpi_disp = chpi_moving_head - np.tile(chpi_static_head, (len(time), 1, 1))
    hpi_disp = np.sqrt((hpi_disp**2).sum(axis=-1))
    # compute velocities
    hpi_velo = chpi_moving_head[1:, :, :] - chpi_moving_head[:-1, :, :]
    hpi_velo = np.sqrt((hpi_velo**2).sum(axis=-1))
    hpi_velo /= np.tile(time[1:] - time[:-1], (n_hpi, 1)).transpose()
    hpi_velo = np.concatenate((np.zeros((1, n_hpi)), hpi_velo), axis=0)

    if disp_thr is not None:
        art_mask = hpi_disp > disp_thr
        annot += _annotations_from_mask(time, art_mask,
                                        'Bad-motion-dist>%0.3f' % disp_thr)
    if velo_thr is not None:
        art_mask = hpi_velo > velo_thr
        annot += _annotations_from_mask(time, art_mask,
                                        'Bad-motion-velo>%0.3f' % velo_thr)

    if gof_thr is not None:
        art_mask = pos[:, 7] <= gof_thr
        annot += _annotations_from_mask(time, art_mask,
                                        'Bad-chpi_gof>%0.3f' % gof_thr)

    tmp = 1000 * hpi_disp.max(axis=0)
    _fmt = '\tHPI00 - %0.1f'
    for i in range(1, n_hpi):
        _fmt += '\n\tHPI%02d' % (i) +' - %0.1f'
    logger.info('CHPI MAX Displacments (mm):')
    logger.info(_fmt % tuple(tmp))
    tmp = 1000 * hpi_velo.max(axis=0)
    logger.info('CHPI Velocity Displacments (mm/sec):')
    logger.info(_fmt % tuple(tmp))

    raw_hpi = None
    if return_stat_raw:
        n_times = len(raw.times)
        # build full time data arrays
        start_idx = raw.time_as_index(time, use_rounding=True)
        end_idx = raw.time_as_index(np.append(time[1:], raw.times[-1]),
                                    use_rounding=True)
        data_pos = np.zeros((2*n_hpi, n_times))
        for t_0, t_1, disp_val, velo_val in zip(start_idx, end_idx,
                                                hpi_disp, hpi_velo):
            t_slice = slice(t_0, t_1)
            data_pos[:n_hpi, t_slice] = np.tile(disp_val, (t_1 - t_0, 1)).T
            data_pos[n_hpi:, t_slice] = np.tile(velo_val, (t_1 - t_0, 1)).T

        ch_names = []
        ch_names_ = []
        for i in range(n_hpi):
            ch_names.append('HPI%02d_disp_pos' % i)
            ch_names_.append('HPI%02d_velo_pos' % i)
        ch_names.extend(ch_names_)

        # build raw object!
        info = create_info(
                        ch_names=ch_names,
                        ch_types=np.repeat('misc', len(ch_names)),
                        sfreq=raw.info['sfreq'])
        raw_hpi = RawArray(data_pos, info)

    return annot, raw_hpi


def plot_artifacts(art_raw, thresholds=dict(), axs=None):

    if axs is None:
        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12.83, 4.66))
    elif len(axs) !=4:
        raise ValueError('please supply 4 axis for plotting')

    _TICKm = np.array([0, 0.5, 1])

    art_data = art_raw.get_data()
    annot = art_raw.annotations

    # plot gfp_normed
    sel = pick_channels(art_raw.ch_names, ['gfp_normed'])
    if len(sel) == 1:
        if annot is not None:
            for on, dur, desc in zip(annot.onset, annot.duration,
                                     annot.description):
                if 'gfp' in desc.lower():
                    axs[0].axvspan(on, on+dur, color='r', alpha=0.2)
        axs[0].plot(art_raw.times, art_data[sel, :].T)
        if thresholds.get('gfp_thresh', None) is not None:
            axs[0].axhline(thresholds['gfp_thresh'], color='r')
            axs[0].set_ylim(top=1.5*thresholds['gfp_thresh'])
            new_ticks = np.round(_TICKm * thresholds['gfp_thresh'], decimals=2)
            axs[0].set_yticks(new_ticks)
        axs[0].set_ylabel('GFP/median')

    # plot displacements...
    sel = pick_channels_regexp(art_raw.ch_names, 'HPI\d+_disp_pos')
    # sel = pick_channels(art_raw.ch_names, ['LPA_disp_pos', 'NAS_disp_pos', 'RPA_disp_pos'])
    if len(sel) >= 1:
        if annot is not None:
            for on, dur, desc in zip(annot.onset, annot.duration,
                                     annot.description):
                if 'motion-dist' in desc.lower():
                    axs[1].axvspan(on, on+dur, color='r', alpha=0.2)
        axs[1].plot(art_raw.times, 100*art_data[sel, :].T)
        if thresholds.get('motion_disp_thresh', None) is not None:
            axs[1].axhline(100*thresholds['motion_disp_thresh'], color='r')
            axs[1].set_ylim(top=1.5*100*thresholds['motion_disp_thresh'])
            new_ticks = np.round(_TICKm * 100*thresholds['motion_disp_thresh'],
                                 decimals=2)
            axs[1].set_yticks(new_ticks)
        axs[1].set_ylabel('hpi disp (cm)')
        # axs[1].legend([art_raw.ch_names[k] for k in sel], loc=1)

    # plot velocities...
    sel = pick_channels_regexp(art_raw.ch_names, 'HPI\d+_velo_pos')
    # sel = pick_channels(art_raw.ch_names, ['LPA_velo_pos', 'NAS_velo_pos', 'RPA_velo_pos'])
    if len(sel) >= 1:
        if annot is not None:
            for on, dur, desc in zip(annot.onset, annot.duration,
                                     annot.description):
                if 'motion-velo' in desc.lower():
                    axs[2].axvspan(on, on+dur, color='r', alpha=0.2)
        axs[2].plot(art_raw.times, 100*art_data[sel, :].T)
        if thresholds.get('motion_velo_thresh', None) is not None:
            axs[2].axhline(100*thresholds['motion_velo_thresh'], color='r')
            axs[2].set_ylim(top=1.5*100*thresholds['motion_velo_thresh'])
            new_ticks = np.round(_TICKm * 100*thresholds['motion_velo_thresh'],
                                 decimals=2)
            axs[2].set_yticks(new_ticks)
        axs[2].set_ylabel('hpi velo (cm/s)')
        # axs[2].legend([art_raw.ch_names[k] for k in sel], loc=1)

    # plot muscle...
    sel = pick_channels(art_raw.ch_names, ['mucsl_score'])
    if len(sel) == 1:
        if annot is not None:
            for on, dur, desc in zip(annot.onset, annot.duration,
                                     annot.description):
                if 'muscle' in desc.lower():
                    axs[3].axvspan(on, on+dur, color='r', alpha=0.2)
        axs[3].plot(art_raw.times, art_data[sel, :].T)
        if thresholds.get('muscle_thresh', None) is not None:
            axs[3].axhline(thresholds['muscle_thresh'], color='r')
            axs[3].set_ylim(top=1.5*thresholds['muscle_thresh'])
            new_ticks = np.round(_TICKm * thresholds['muscle_thresh'], decimals=2)
            axs[3].set_yticks(new_ticks)

        axs[3].set_ylabel('Muscle Stat')
        axs[3].set_xlabel('Time (s)')

    for ax in axs:
        ax.set_xlim(art_raw.times[0], art_raw.times[-1])


###################################################
def _dev_head_objective(x, chpi_head, weights, chpi_locs_dev):
    """Compute objective function."""
    chpi_locs_head = _apply_quat(x, chpi_locs_dev, move=True)
    err = (chpi_head - np.tile(chpi_locs_head, (chpi_head.shape[0], 1, 1)))**2
    return np.dot(weights, err.sum(-1).sum(-1))


def _fit_dev_head(init_dev_head_t, time, quats, chpi_locs_dev,
                  gof, gof_thresh):
    """Fit rotation and translation (quaternion) parameters for cHPI coils."""
    weights = np.append(time[1:] - time[:-1], 0)

    # zero contibution for poor fits...
    weights[gof < gof_thresh] = 0
    weights /= sum(weights)

    chpi_head = np.array([_apply_quat(quat, chpi_locs_dev, move=True)
                          for quat in quats])
    print("CHPI_HEAD\n", chpi_head)
    print("CHPI_DEV\n", chpi_locs_dev[0])

    n_hpi = chpi_locs_dev.shape[0]

    chpi_head_flat = chpi_head.reshape(-1, 3 * n_hpi)
    tmp_disp = ((chpi_head_flat -
                 np.average(chpi_head_flat, axis=0,
                            weights=weights)) ** 2).sum(axis=1)

    init_dev_head_t = _quaternion_align(init_dev_head_t['from'],
                                        init_dev_head_t['to'], chpi_locs_dev,
                                        chpi_head[tmp_disp.argmin(), :, :])
    print ('Index and time - min')
    print(tmp_disp.argmin(), time[tmp_disp.argmin()])

    print ('Moving Points at min')
    print (chpi_head[tmp_disp.argmin(), :, :])

    print ('New trans')
    print (init_dev_head_t)

    print ('New head points')
    print (apply_trans(init_dev_head_t, chpi_locs_dev))

    dev_head_t = init_dev_head_t
    result = 0
    denom = 1
    return dev_head_t, 1. - result / denom


def compute_dev_head_trans(info, pos, gof_thresh=0.99):
    """Compute dev_head transform to minimize continuous displacements."""
    info = info.copy()
    chpi_locs_dev = sorted([d for d in info['hpi_results'][-1]
                            ['dig_points']], key=lambda x: x['ident'])
    chpi_locs_dev = np.array([d['r'] for d in chpi_locs_dev])

    dev_head_t, g = _fit_dev_head(info['dev_head_t'], pos[:, 0], pos[:, 1:7],
                                  chpi_locs_dev, pos[:, 7], gof_thresh)
    logger.info('computing new dev_head_t yeiled a GOF of %0.3f' % g)
    return dev_head_t


###################################################
def write_annotations_to_ctf(annot, sfreq, ds_fname, seg_bname='bad.segments'):
    """Write out annotations to ds file."""
    res4 = _read_res4(ds_fname)
    start_time = -res4['pre_trig_pts'] / float(sfreq)
    out_fname = op.join(ds_fname, seg_bname)
    onsets = start_time + annot.onset

    with open(out_fname, 'w') as fid:
        for on, dur, desc in zip(onsets, annot.duration, annot.description):
            fid.write('%s\t\t%f\t\t%f\t\t\n' % (desc, on, on + dur))


def read_annotations_besa_evt(evt_fname, time_0=0.0):
    """Read in evt file as annotations."""
    seg_info = dict()
    with open(evt_fname, 'r') as fid:
        fid.readline()
        for line in fid.readlines():
            parts = line.strip().split('\t')
            code = parts[1]
            time = time_0 + 1e-6 * float(parts[0].strip())
            if code not in seg_info:
                seg_info[code] = []
            seg_info[code].append(time)

    # build Annotations
    onsets = []
    durations = []
    desc = []
    for code, t_list in seg_info.iteritems():
        it = iter(t_list)
        for t_start in it:
            onsets.append(t_start)
            durations.append(next(it) - t_start)
            desc.append('besa_%s' % code)
    return Annotations(onsets, durations, desc)
