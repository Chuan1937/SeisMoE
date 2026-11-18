import os
import csv
import math
import time
import h5py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
import seisbench.models as sbm
from obspy.signal.trigger import trigger_onset
import multiprocessing as mp

import numpy as np
from obspy.signal.trigger import trigger_onset  # Not strictly needed, but kept for consistency
from datetime import datetime, timedelta

def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):
    """
    Identical to your existing _detect_peaks. 
    """
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    dx = x[1:] - x[:-1]

    # Handle NaNs
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf

    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]

    ind = np.unique(np.hstack((ine, ire, ife)))

    # Filter out neighbors of NaNs
    if ind.size and indnan.size:
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # Remove edge peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # Minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # Threshold
    if ind.size and threshold > 0:
        dx_min = np.min(np.vstack([x[ind] - x[ind-1], x[ind] - x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx_min < threshold)[0])
    # Enforce mpd
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                idel = idel | (ind >= ind[i]-mpd) & (ind <= ind[i]+mpd) & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0
        ind = np.sort(ind[~idel])
    return ind


def phasenet_picker(P_threshold, S_threshold, p_array, s_array):
    """
    Find P picks above `P_threshold` and S picks above `S_threshold`.
    Then match each P to the first S that arrives within 60 seconds.
    
    Returns a dict of the form:
        {
          p_index: (s_index, p_prob, s_prob)
          ...
        }
    where p_index, s_index are sample indices in [0, N).
    """
    p_inds = _detect_peaks(p_array, mph=P_threshold, mpd=1)
    s_inds = _detect_peaks(s_array, mph=S_threshold, mpd=1)

    matches = {}
    used_s = set()

    for p_idx in p_inds:
        p_time_sec = p_idx / 100.0
        p_prob = p_array[p_idx]

        # find first S pick that arrives within 60s after p
        s_match = None
        for s_idx in s_inds:
            if s_idx in used_s:
                continue
            s_time_sec = s_idx / 100.0
            if 0 < (s_time_sec - p_time_sec) < 60:
                # match found
                s_match = s_idx
                used_s.add(s_idx)
                break
        
        if s_match is not None:
            s_prob = s_array[s_match]
            matches[p_idx] = (s_match, p_prob, s_prob)
        else:
            # If you want to keep P-only picks, you can do so here
            # matches[p_idx] = (None, p_prob, None)
            pass

    return matches


import math
import numpy as np

def _get_snr(data, pick_index, window=200):
    """
    data: 1D array from a single channel, or 
          you can pass the trace magnitude if you want.
    pick_index: sample index
    window: half-window length around pick to define noise & signal windows.
    """
    snr = None
    if pick_index is not None:
        try:
            pick_index = int(pick_index)
            nstart = max(0, pick_index - window)
            nend   = pick_index
            sstart = pick_index
            send   = min(pick_index + window, len(data))
            noise = data[nstart:nend]
            signal = data[sstart:send]

            # Simple robust ratio
            if len(noise) > 0 and len(signal) > 0:
                snr_val = (np.percentile(signal, 95) / np.percentile(noise, 95))**2
                snr = round(10 * math.log10(snr_val), 1)
        except:
            pass
    return snr


def phasenet_prediction_writer(
    dataset_h5,
    trace_name,
    p_idx,
    s_idx,
    p_prob,
    s_prob,
    csv_writer
):
    """
    Write a single row for a matched (P, S) pick to CSV.
    
    dataset_h5: the open HDF5 handle so we can access attrs for `trace_name`
    trace_name: name of the group in HDF5
    p_idx, s_idx: sample indices
    p_prob, s_prob: probabilities
    csv_writer: an open csv writer

    We assume each trace has attributes like:
       dataset.attrs["trace_name"],
       dataset.attrs["receiver_code"], etc.
    We'll mark event start time as the P pick, event end time as the S pick, etc.
    """
    dset = dataset_h5['data'][trace_name]
    # read basic metadata
    tr_name = dset.attrs["trace_name"]
    station_name = dset.attrs["receiver_code"]
    station_lat = dset.attrs["receiver_latitude"]
    station_lon = dset.attrs["receiver_longitude"]
    station_elv = dset.attrs["receiver_elevation_m"]
    net_code = dset.attrs["network_code"]
    instrument_type = tr_name.split('_')[2] if len(tr_name.split('_')) > 2 else '??'
    
    start_time_str = dset.attrs["trace_start_time"]
    try:
        start_dt = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S.%f')
    except:
        start_dt = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')

    # Convert sample indices to actual times (assuming 100 Hz)
    p_time = start_dt + timedelta(seconds=p_idx / 100.0) if p_idx is not None else None
    s_time = start_dt + timedelta(seconds=s_idx / 100.0) if s_idx is not None else None

    # Simple approach: treat "event_start_time" as the P pick, "event_end_time" as the S pick
    event_start = p_time
    event_end   = s_time

    # For SNR, let's say channel 0 for P picks, channel 1 for S picks
    data = np.array(dset, dtype=np.float32)
    # if you prefer to do SNR on vertical or something else, adjust as needed:
    # e.g. data[:,0] is the first channel
    p_snr = _get_snr(data[:,0], p_idx, 200) if p_idx else None
    s_snr = _get_snr(data[:,1], s_idx, 200) if s_idx else None

    csv_writer.writerow([
        tr_name,
        net_code,
        station_name,
        instrument_type,
        station_lat,
        station_lon,
        station_elv,
        event_start.strftime('%Y-%m-%d %H:%M:%S.%f') if event_start else "",
        event_end.strftime('%Y-%m-%d %H:%M:%S.%f') if event_end else "",
        "",                # detection_probability (Not used in PhaseNet)
        "",                # detection_uncertainty  (Not used in PhaseNet)
        p_time.strftime('%Y-%m-%d %H:%M:%S.%f') if p_time else "",
        round(p_prob,3) if p_prob else "",
        "",                # p_uncertainty (Not used in basic PhaseNet)
        p_snr if p_snr else "",
        s_time.strftime('%Y-%m-%d %H:%M:%S.%f') if s_time else "",
        round(s_prob,3) if s_prob else "",
        "",                # s_uncertainty
        s_snr if s_snr else ""
    ])
