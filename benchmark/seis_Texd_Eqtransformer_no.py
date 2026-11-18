import os
import time
import csv
import h5py
import torch
import numpy as np
import pandas as pd
import random
import gc
import sys
import multiprocessing as mp
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
import seisbench.models as sbm
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from obspy.signal.trigger import trigger_onset

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

def prediction_writer(trace_name, predict_writer, csvPr, matches, snr, detection_memory,input_csv):
    """
    Writes each event row (including partial P or S).
    If P_idx is None => p_time, p_prob, p_unc, p_snr = empty.
    If S_idx is None => s_time, s_prob, s_unc, s_snr = empty.
    """

    df=pd.read_csv(input_csv)
    dataset=df[df['trace_name']==trace_name]

    trace_name = dataset["trace_name"].values
    station_name = dataset["station_code"].values
    station_lat = dataset["station_latitude_deg"].values
    station_lon = dataset["station_longitude_deg"].values
    station_elv = dataset["station_elevation_m"].values
    start_time = dataset["source_origin_time"].values
    network_name = dataset["station_code"].values


    start_time= pd.to_datetime(start_time).strftime("%Y/%m/%d %H:%M:%S")

    start_time=pd.to_datetime(start_time.values)
    
    # trace_name = dataset.attrs["trace_name"]
    # station_name = dataset.attrs["receiver_code"]
    # station_lat = dataset.attrs["receiver_latitude"]
    # station_lon = dataset.attrs["receiver_longitude"]
    # station_elv = dataset.attrs["receiver_elevation_m"]
    # start_time = dataset.attrs["trace_start_time"]
    # network_name = dataset.attrs["network_code"]


    # Instrument type parsing (example)
    # fallback if not found
    # parts = trace_name.split('$')
    # if len(parts) > 2:
    #     instrument_type = parts[2]
    # else:
    #     instrument_type = "??"

    instrument_type = "??"

    # # Convert time string
    # try:
    #     start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')
    # except:
    #     start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')

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
        ev_strt=ev_strt[0]
        ev_end  = start_time + timedelta(seconds=match_value[0] / 100.0)
        ev_end=ev_end[0]
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
            p_time=p_time[0]
        
        # S picks
        s_idx  = match_value[6]
        s_prob = match_value[7] if match_value[7] else None
        s_unc  = match_value[8] if match_value[8] else None

        s_time = None
        if s_idx is not None:
            s_time = start_time + timedelta(seconds=s_idx/100.0)
            s_time=s_time[0]

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



###############################################################
# 2) 改写 Dataset，不在 __init__ 持有 h5py.File
###############################################################
class SeismicDataset(Dataset):
    def __init__(self, hdf5_file, trace_names, normalization_mode='std'):
        self.hdf5_path = hdf5_file
        self.trace_names = trace_names
        self.normalization_mode = normalization_mode

    def __len__(self):
        return len(self.trace_names)

    def __getitem__(self, idx):
        trace_name = self.trace_names[idx]
        with h5py.File(self.hdf5_path, 'r') as f:
            data = np.array(f['data'][trace_name], dtype=np.float32)

        if self.normalization_mode == 'std':
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-10)
        elif self.normalization_mode == 'max':
            data = data / (np.max(np.abs(data), axis=0) + 1e-10)

        # EQTransformer 模型需要 (3, N)
        # data = data.T
        data_torch = torch.from_numpy(data)
        return {
            'input': data_torch,
            'trace_name': trace_name
        }

###############################################################
# 3) 多进程并行后处理（含进度条）
###############################################################
def process_sublist(
    sublist,
    hdf5_file,
    prob_dic,
    detection_threshold,
    P_threshold,
    S_threshold,
    partial_csv_prefix
):
    """
    每个子进程的工作：
      - 读取 sublist: [(trace_name, idx_array), ...]
      - 对每条记录执行: (1) 读取波形  (2) picker(...) (3) SNR + 2秒内去重 + 写 partial CSV
      - 通过 tqdm 显示子进程进度, desc 显示 PID.
    """
    import h5py
    partial_csv = partial_csv_prefix + f".pid_{os.getpid()}.csv"

    # 写 CSV 表头
    with open(partial_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'file_name', 'network', 'station', 'instrument_type',
            'station_lat', 'station_lon', 'station_elv',
            'event_start_time', 'event_end_time',
            'detection_probability', 'detection_uncertainty',
            'p_arrival_time', 'p_probability', 'p_uncertainty', 'p_snr',
            's_arrival_time', 's_probability', 's_uncertainty', 's_snr'
        ])

    detection_memory_local = []

    with h5py.File(hdf5_file, 'r') as h5, open(partial_csv, 'a', newline='') as f:
        writer = csv.writer(f)

        # 加进度条
        for (trace_name, idx_ar) in tqdm(
            sublist,
            desc=f"Worker {os.getpid()}",
            total=len(sublist),
            leave=True
        ):
            dat = np.array(h5['data'][trace_name])
            # 归一化
            dat = (dat - np.mean(dat, axis=0)) / (np.std(dat, axis=0) + 1e-10)

            # 取该记录预测好的 (D, P, S) 概率
            dd_mean = prob_dic['DD_mean'][idx_ar]
            pp_mean = prob_dic['PP_mean'][idx_ar]
            ss_mean = prob_dic['SS_mean'][idx_ar]
            dd_std  = prob_dic['DD_std'][idx_ar]
            pp_std  = prob_dic['PP_std'][idx_ar]
            ss_std  = prob_dic['SS_std'][idx_ar]

            # 调用 picker
            matches, _, _ = picker(
                detection_threshold, P_threshold, S_threshold,
                dd_mean, pp_mean, ss_mean,
                dd_std, pp_std, ss_std
            )
            # 只处理第一个 match
            if len(matches) >= 1:
                first_key = list(matches)[0]
                if matches[first_key][3] and matches[first_key][6]:
                    if (matches[first_key][6] - matches[first_key][3]) < 60 * 100:
                        # 计算 SNR
                        snr_val = [
                            _get_snr(dat, matches[first_key][3], window=100),
                            _get_snr(dat, matches[first_key][6], window=100)
                        ]
                        # dataset_trace = h5['data'][trace_name]
                        detection_memory_local = prediction_writer(
                            trace_name, writer, f, matches, snr_val,
                            detection_memory_local,input_csv=r'DATA/metadata_no_noise.csv'
                        )

    return partial_csv

def process_predictions_parallel(
    batch_results,
    hdf5_file,
    prob_dic,
    detection_threshold,
    P_threshold,
    S_threshold,
    partial_csv_prefix
):
    """
    并行调度：将 batch_results 分块给多个进程，写 partial CSV
    """
    import math
    from functools import partial

    n_proc = min(8, mp.cpu_count())  # 并行进程数
    chunk_size = math.ceil(len(batch_results) / n_proc)
    chunks = [batch_results[i:i+chunk_size] for i in range(0, len(batch_results), chunk_size)]

    with mp.Pool(n_proc) as pool:
        func = partial(
            process_sublist,
            hdf5_file=hdf5_file,
            prob_dic=prob_dic,
            detection_threshold=detection_threshold,
            P_threshold=P_threshold,
            S_threshold=S_threshold,
            partial_csv_prefix=partial_csv_prefix
        )
        partial_csv_list = pool.map(func, chunks)

    return partial_csv_list

def final_deduplicate(partial_csv_list, final_csv_path):
    """
    合并所有 partial CSV，然后做一次全局 2 秒去重
    """
    dfs = []
    for pcsv in partial_csv_list:
        if os.path.exists(pcsv):
            dfs.append(pd.read_csv(pcsv))
    if len(dfs) == 0:
        pd.DataFrame().to_csv(final_csv_path, index=False)
        return 0

    big_df = pd.concat(dfs, ignore_index=True)
    # 按 event_start_time 排序
    big_df['event_start_time'] = pd.to_datetime(big_df['event_start_time'], errors='coerce')
    big_df.sort_values(by='event_start_time', inplace=True)

    deduped_rows = []
    last_time = None
    for _, row in big_df.iterrows():
        cur_time = row['event_start_time']
        if pd.isnull(cur_time):
            deduped_rows.append(row)
            continue
        if last_time is None:
            deduped_rows.append(row)
            last_time = cur_time
        else:
            delta_sec = (cur_time - last_time).total_seconds()
            if delta_sec >= 2:
                deduped_rows.append(row)
                last_time = cur_time
            else:
                pass

    final_df = pd.DataFrame(deduped_rows)
    final_df.to_csv(final_csv_path, index=False)
    return len(final_df)

###############################################################
# 4) 主函数 eqt_prediction 并行后处理 (带CPU多进程筛选+进度条)
###############################################################
def eqt_prediction_parallel(
    input_csv,
    input_hdf5,
    store_dir,
    detection_threshold=0.5,
    P_threshold=0.3,
    S_threshold=0.3,
    batch_size=20,
    device=None,
    random_seed=42,
    ratio=0.1
):
    """
    1) 用 EQTransformer 做推理 => 得到 prob_dic
    2) 构造 batch_results => 多进程并行 CPU 后处理 => partial CSV + 进度条
    3) 最后合并 + 2秒去重 => final CSV
    """
    start_time = time.time()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}')
    # ====== (A) 加载 EQTransformer ======
    model = sbm.EQTransformer.from_pretrained("stead").to(device)
    model.eval()

    # ====== (B) 随机抽样
    df_csv = pd.read_csv(input_csv, low_memory=False)
    full_list = df_csv["trace_name"].tolist()
    random.seed(random_seed)
    total_samples = len(full_list)
    sample_size = int(total_samples * ratio)
    predict_list = random.sample(full_list, sample_size)
    print(f"Total {total_samples} traces, sample => {len(predict_list)} for eqt_prediction")

    # ====== (C) GPU/CPU 推理 => 得到 (D, P, S)
    dataset = SeismicDataset(input_hdf5, predict_list, normalization_mode='std')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    all_det, all_p, all_s = [], [], []
    print("Start predicting (EQTransformer) ...")
    with torch.no_grad():
        for batch in tqdm(loader):
            inputs = batch['input'].float().to(device)
            outputs = model(inputs)  # => [det, p, s], each shape (B, N)
            all_det.append(outputs[0].cpu().numpy())
            all_p.append(outputs[1].cpu().numpy())
            all_s.append(outputs[2].cpu().numpy())

    all_det = np.concatenate(all_det, axis=0)
    all_p   = np.concatenate(all_p,   axis=0)
    all_s   = np.concatenate(all_s,   axis=0)

    prob_dic = {
        'DD_mean': all_det,
        'PP_mean': all_p,
        'SS_mean': all_s,
        'DD_std': np.zeros_like(all_det),
        'PP_std': np.zeros_like(all_p),
        'SS_std': np.zeros_like(all_s),
    }

    del all_det, all_p, all_s
    gc.collect()

    # ====== (D) 并行后处理
    # 构造 batch_results = [(trace_name, idx), ...]
    batch_results = [(tn, i) for i, tn in enumerate(predict_list)]

    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    partial_csv_prefix = os.path.join(store_dir, f"STEAD_EQTransformer_Predict_{random_seed}")

    print("Start parallel post-processing (with tqdm for each worker) ...")
    partial_csv_list = process_predictions_parallel(
        batch_results=batch_results,
        hdf5_file=input_hdf5,
        prob_dic=prob_dic,
        detection_threshold=detection_threshold,
        P_threshold=P_threshold,
        S_threshold=S_threshold,
        partial_csv_prefix=partial_csv_prefix
    )

    # ====== (E) 最终合并 + 2秒去重
    final_csv_path = partial_csv_prefix + "_final.csv"
    final_count = final_deduplicate(partial_csv_list, final_csv_path)

    # ====== (F) 打印统计
    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"\n*** Finished eqt_prediction & parallel post-processing in {int(hours)}h {int(minutes)}m {round(seconds, 2)}s.")
    print(f"*** Final deduplicated events: {final_count}")
    print(f"*** CSV => {final_csv_path}")
    sys.exit(0)

#########################################################
# 测试入口
#########################################################
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    input_csv = "/mnt/d/AI_Seismic_Data/txed/Texd.csv"
    input_hdf5 = "DATA/waveforms_all_nonoise.hdf5"
    store_dir = "/home/yuan/code/SeisMoE/data/test"

    eqt_prediction_parallel(
        input_csv=input_csv,
        input_hdf5=input_hdf5,
        store_dir=store_dir,
        detection_threshold=0.1,
        P_threshold=0.1,
        S_threshold=0.1,
        batch_size=200,
        device=None,     # CPU或GPU均可
        random_seed=81,
        ratio=0.001
    )
