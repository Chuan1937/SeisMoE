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

def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):
    """
    Identical to your existing version. 
    Finds indices of peaks in 1D array x.
    """
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    dx = x[1:] - x[:-1]
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
    if ind.size and indnan.size:
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    if ind.size and threshold > 0:
        dx_min = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx_min < threshold)[0])
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0
        ind = np.sort(ind[~idel])
    return ind

def picker(detection_threshold, P_threshold, S_threshold, 
           yh1, yh2, yh3, 
           yh1_std, yh2_std, yh3_std,
           spt=None, sst=None):
    """
    Modified DPS picker that can handle partial picks:
     - detection from yh1
     - P picks from yh2
     - S picks from yh3
    Stores None if no P or no S pick is found in that event window.
    """
    detection = trigger_onset(yh1, detection_threshold, detection_threshold)

    pp_arr = _detect_peaks(yh2, mph=P_threshold, mpd=1)
    ss_arr = _detect_peaks(yh3, mph=S_threshold, mpd=1)

    P_PICKS = {}
    S_PICKS = {}
    EVENTS = {}
    matches = {}
    pick_errors = {}

    # Build P_PICKS dict
    if len(pp_arr) > 0:
        P_uncertainty = None
        for pick in pp_arr:
            if pick is not None:
                P_prob = np.round(yh2[int(pick)], 3)
                P_PICKS.update({pick: [P_prob, P_uncertainty]})

    # Build S_PICKS dict
    if len(ss_arr) > 0:
        S_uncertainty = None
        for pick in ss_arr:
            if pick is not None:
                S_prob = np.round(yh3[int(pick)], 3)
                S_PICKS.update({pick: [S_prob, S_uncertainty]})

    # Build event windows from detection (yh1)
    if len(detection) > 0:
        D_uncertainty = None
        for ev in detection:
            bg = ev[0]
            ed = ev[1]
            D_prob = np.mean(yh1[bg:ed])
            D_prob = np.round(D_prob, 3)
            EVENTS.update({bg: [D_prob, D_uncertainty, ed]})

    # For each detection window, gather possible P, S picks
    for ev in EVENTS:
        bg = ev
        ed = EVENTS[ev][2]
        event_prob = EVENTS[ev][0]
        event_unc  = EVENTS[ev][1]

        if (ed - bg) < 10:
            # detection window too short, skip
            continue

        # Gather candidate S picks if inside [bg, ed]
        candidate_Ss = {}
        for Ss, S_val in S_PICKS.items():
            if Ss >= bg and Ss <= ed:
                candidate_Ss[Ss] = S_val
        # if none found, fill with (None, None)
        if len(candidate_Ss) == 0:
            candidate_Ss = {None: [None, None]}
        # If multiple S picks in that window, keep the earliest or highest S-prob
        if len(candidate_Ss) > 1:
            # Example: keep the highest-prob S
            best_S = max(candidate_Ss.keys(), key=lambda x: candidate_Ss[x][0] if x else 0)
            candidate_Ss = {best_S: candidate_Ss[best_S]}

        # Gather candidate P picks
        candidate_Ps = {}
        for Ps, P_val in P_PICKS.items():
            # If we found an S, we only want P that occurs a bit before S
            the_only_S = list(candidate_Ss)[0]
            if the_only_S is not None:
                if Ps >= (bg - 100) and Ps < (the_only_S - 10):
                    candidate_Ps[Ps] = P_val
            else:
                # if no S, allow P in [bg-100, ed]
                if Ps >= (bg - 100) and Ps <= ed:
                    candidate_Ps[Ps] = P_val
        if len(candidate_Ps) == 0:
            candidate_Ps = {None: [None, None]}
        # If multiple P picks, keep e.g. the one with highest P-prob
        if len(candidate_Ps) > 1:
            best_P = max(candidate_Ps.keys(), key=lambda x: candidate_Ps[x][0] if x else 0)
            candidate_Ps = {best_P: candidate_Ps[best_P]}

        # Now combine them. We only record if at least one is not None.
        # That is: "if we found P or S" => record
        P_idx = list(candidate_Ps)[0]
        P_prob, P_unc = candidate_Ps[P_idx]
        S_idx = list(candidate_Ss)[0]
        S_prob, S_unc = candidate_Ss[S_idx]

        if (P_idx is not None) or (S_idx is not None):
            # We record the event
            matches[bg] = [
                ed,          # 0
                event_prob,  # 1 (detection probability)
                event_unc,   # 2 (detection uncertainty)
                P_idx,       # 3
                P_prob,      # 4
                P_unc,       # 5
                S_idx,       # 6
                S_prob,      # 7
                S_unc        # 8
            ]

        # (Optional) if you want to track picking errors:
        # pick_errors[bg] = [some_value, etc...]

    return matches, pick_errors, yh3  # returning yh3 is your original design

def _get_snr(data, pick_index, window=200):
    """
    Calculate SNR around pick_index using a +/- 'window' approach.
    If pick_index is None, returns None.
    """
    snr = None
    if pick_index is not None:
        try:
            pick_index = int(pick_index)
            if pick_index >= window and (pick_index + window) < len(data):
                noise = data[pick_index - window: pick_index]
                signal = data[pick_index: pick_index + window]
            elif pick_index < window:
                # If pick is too close to start, reduce the noise window
                noise = data[0: pick_index]
                signal = data[pick_index: pick_index + window]
            else:
                # If pick + window is beyond end, reduce the signal window
                noise = data[pick_index - window: pick_index]
                signal = data[pick_index: ]
            if len(noise) > 0 and len(signal) > 0:
                snr_val = (np.percentile(signal, 95) / np.percentile(noise, 95)) ** 2
                snr = round(10 * math.log10(snr_val), 1)
        except:
            pass
    return snr

def prediction_writer(dataset, predict_writer, csvPr, matches, snr, detection_memory):
    """
    Writes each event row (including partial P or S).
    If P_idx is None => p_time, p_prob, p_unc, p_snr = empty.
    If S_idx is None => s_time, s_prob, s_unc, s_snr = empty.
    """
    trace_name = dataset.attrs["trace_name"]
    station_name = dataset.attrs["receiver_code"]
    station_lat = dataset.attrs["receiver_latitude"]
    station_lon = dataset.attrs["receiver_longitude"]
    station_elv = dataset.attrs["receiver_elevation_m"]
    start_time = dataset.attrs["trace_start_time"]
    network_name = dataset.attrs["network_code"]


    # Instrument type parsing (example)
    # fallback if not found
    parts = trace_name.split('_')
    if len(parts) > 2:
        instrument_type = parts[2]
    else:
        instrument_type = "??"

    # Convert time string
    try:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')
    except:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')

    def _date_convertor(r):
        if r is None:
            return ""
        elif isinstance(r, datetime):
            return r.strftime('%Y-%m-%d %H:%M:%S.%f')
        elif isinstance(r, str):
            return r  # or parse it if needed
        else:
            return str(r)

    for match, match_value in matches.items():
        ev_strt = start_time + timedelta(seconds=match / 100.0)
        ev_end  = start_time + timedelta(seconds=match_value[0] / 100.0)
        
        # Avoid writing duplicates if event_start is within 2s of a previously written event
        doublet = [st for st in detection_memory if abs((st - ev_strt).total_seconds()) < 2]
        if len(doublet) > 0:
            continue

        # detection probability / uncertainty
        det_prob = round(match_value[1], 2) if match_value[1] is not None else None
        det_unc  = round(match_value[2], 2) if match_value[2] is not None else None

        # P picks
        p_idx  = match_value[3]
        p_prob = match_value[4] if match_value[4] else None
        p_unc  = match_value[5] if match_value[5] else None

        p_time = None
        if p_idx is not None:
            p_time = start_time + timedelta(seconds=p_idx/100.0)
        
        # S picks
        s_idx  = match_value[6]
        s_prob = match_value[7] if match_value[7] else None
        s_unc  = match_value[8] if match_value[8] else None

        s_time = None
        if s_idx is not None:
            s_time = start_time + timedelta(seconds=s_idx/100.0)

        # SNR array e.g. [p_snr, s_snr] from outside
        # snr[0], snr[1] if we used "first picks" or etc.
        # But typically you'd compute them for each match, e.g.:
        #  p_snr = snr_for_p, s_snr = snr_for_s
        # For your code, you had something like:
        # snr = [ _get_snr(data, p_idx, 100), _get_snr(data, s_idx, 100) ]
        # We'll just trust you're passing that in.
        p_snr = snr[0] if snr and len(snr) > 0 else None
        s_snr = snr[1] if snr and len(snr) > 1 else None

        predict_writer.writerow([
            trace_name,
            network_name,
            station_name,
            instrument_type,
            station_lat,
            station_lon,
            station_elv,
            _date_convertor(ev_strt),
            _date_convertor(ev_end),
            det_prob if det_prob else "",
            det_unc  if det_unc  else "",
            _date_convertor(p_time),
            p_prob if p_prob else "",
            p_unc  if p_unc  else "",
            p_snr  if p_snr else "",
            _date_convertor(s_time),
            s_prob if s_prob else "",
            s_unc  if s_unc else "",
            s_snr  if s_snr else ""
        ])
        csvPr.flush()
        detection_memory.append(ev_strt)

    return detection_memory
