#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import csv
import math
import time
import h5py
import torch
import numpy as np
import pandas as pd
import random
import gc
import multiprocessing as mp
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import seisbench.models as sbm

###################################
# 1) 公共函数：phasenet_picker, etc.
###################################


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
                idel = idel | ((ind >= ind[i]-mpd) & (ind <= ind[i]+mpd) 
                               & (x[ind[i]] > x[ind] if kpsh else True))
                idel[i] = 0
        ind = np.sort(ind[~idel])
    return ind

def phasenet_picker(P_threshold, S_threshold, p_array, s_array):
    """
    修改后:
      只保留 p_array 中概率最高的那个 P 拾取，
      在其后 60s (可自行改) 内寻找第一个符合条件的 S 拾取。
      若没找到 S，则只输出 P。
      这样对于一个 trace 只会输出一个事件。
    
    返回形式:
        { best_p_idx: (best_s_idx or None, best_p_prob, best_s_prob or None) }
    """
    # 1) 找所有 P 的峰值
    p_inds = _detect_peaks(p_array, mph=P_threshold, mpd=1)
    if len(p_inds) == 0:
        # 没有任何 P，返回空
        return {}
    # 2) 找到概率最高的 P
    best_p_idx = None
    best_p_prob = -999
    for p_idx in p_inds:
        prob = p_array[p_idx]
        if prob > best_p_prob:
            best_p_idx = p_idx
            best_p_prob = prob

    # 3) 在 60s 内找第一个符合门槛的 S
    s_inds = _detect_peaks(s_array, mph=S_threshold, mpd=1)
    best_s_idx = None
    best_s_prob = None

    p_time_sec = best_p_idx / 100.0
    for s_idx in s_inds:
        s_time_sec = s_idx / 100.0
        if 0 < (s_time_sec - p_time_sec) < 60:
            best_s_idx = s_idx
            best_s_prob = s_array[s_idx]
            break  # 只取第一个符合的 S

    # 4) 组装返回
    matches = {}
    if best_s_idx is not None:
        matches[best_p_idx] = (best_s_idx, best_p_prob, best_s_prob)
    else:
        # 如果想只保留「P+S 组合」则不写
        # 但此处我们保留仅 P 的结果
        matches[best_p_idx] = (None, best_p_prob, None)

    return matches


def _get_snr(data, pick_index, window=200):
    """
    计算 SNR
    """
    snr = None
    if pick_index is not None:
        try:
            pick_index = int(pick_index)
            nstart = max(0, pick_index - window)
            nend   = pick_index
            sstart = pick_index
            send   = min(pick_index + window, len(data))
            noise  = data[nstart:nend]
            signal = data[sstart:send]
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
    写入一行 (P, S) 到 CSV. 若某个为空则写空。
    """
    dset = dataset_h5['data'][trace_name]
    tr_name = dset.attrs["trace_name"]
    station_name = dset.attrs["receiver_code"]
    station_lat  = dset.attrs["receiver_latitude"]
    station_lon  = dset.attrs["receiver_longitude"]
    station_elv  = dset.attrs["receiver_elevation_m"]
    net_code     = dset.attrs["network_code"]
    parts = tr_name.split('_')
    instrument_type = parts[2] if len(parts) > 2 else '??'

    start_time_str = dset.attrs["trace_start_time"]
    try:
        start_dt = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S.%f')
    except:
        start_dt = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')

    # 采样率假设 100 Hz
    p_time = (start_dt + timedelta(seconds=p_idx/100.0)) if p_idx is not None else None
    s_time = (start_dt + timedelta(seconds=s_idx/100.0)) if s_idx is not None else None

    event_start = p_time
    event_end   = s_time

    data_3c = np.array(dset, dtype=np.float32)
    p_snr = _get_snr(data_3c[:,0], p_idx, 200) if p_idx else None
    s_snr = _get_snr(data_3c[:,1], s_idx, 200) if s_idx else None

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
        "",  # detection_probability
        "",  # detection_uncertainty
        p_time.strftime('%Y-%m-%d %H:%M:%S.%f') if p_time else "",
        round(p_prob,3) if p_prob else "",
        "",  # p_uncertainty
        p_snr if p_snr else "",
        s_time.strftime('%Y-%m-%d %H:%M:%S.%f') if s_time else "",
        round(s_prob,3) if s_prob else "",
        "",  # s_uncertainty
        s_snr if s_snr else ""
    ])

###################################
# 2) Dataset 读取: PhaseNet inference
###################################
class SeismicDataset(Dataset):
    """
    读取 HDF5，不在 __init__ 里保存 h5py.File，
    每次 __getitem__ 临时打开 => 支持多进程 DataLoader
    """
    def __init__(self, hdf5_file, trace_names, normalization_mode='std'):
        self.hdf5_path= hdf5_file
        self.trace_names= trace_names
        self.normalization_mode= normalization_mode

    def __len__(self):
        return len(self.trace_names)

    def __getitem__(self, idx):
        trace_name= self.trace_names[idx]
        with h5py.File(self.hdf5_path,'r') as f:
            data= np.array(f['data'][trace_name], dtype=np.float32)
        # 归一化
        if self.normalization_mode=='std':
            data= (data-np.mean(data,axis=0))/ (np.std(data,axis=0)+1e-10)
        elif self.normalization_mode=='max':
            data= data / (np.max(np.abs(data),axis=0)+1e-10)
        # (3,N) for PhaseNet
        data= data.T
        return {"trace_name": trace_name, "data": torch.from_numpy(data)}

###################################
# 3) 并行后处理
###################################
def process_sublist(
    sublist,   # [(trace_name, p_array, s_array), ...]
    hdf5_file,
    p_threshold,
    s_threshold,
    partial_prefix
):
    """
    每个子进程处理 sublist, 并加上tqdm进度
    """
    from tqdm import tqdm

    out_csv= partial_prefix+f".pid_{os.getpid()}.csv"
    with open(out_csv,'w', newline='') as f:
        w= csv.writer(f)
        w.writerow([
            'file_name','network','station','instrument_type',
            'station_lat','station_lon','station_elv',
            'event_start_time','event_end_time',
            'detection_probability','detection_uncertainty',
            'p_arrival_time','p_probability','p_uncertainty','p_snr',
            's_arrival_time','s_probability','s_uncertainty','s_snr'
        ])

    with h5py.File(hdf5_file,'r') as h5, open(out_csv,'a',newline='') as f:
        writer= csv.writer(f)
        # detection_memory= [] # 如果只需要去掉同进程 2秒重复,可加; 但你要全局2秒 =>后面deduplicate
        for (trace_name, p_arr, s_arr) in tqdm(sublist, desc=f"Worker {os.getpid()}", total=len(sublist)):
            # 直接调用 phasenet_picker
            matches= phasenet_picker(p_threshold, s_threshold, p_arr, s_arr)
            # matches => { p_idx: (s_idx, p_prob, s_prob)}
            # 只会有一个 key
            for p_idx, val in matches.items():
                s_idx, p_prob, s_prob= val
                phasenet_prediction_writer(
                    h5, trace_name,
                    p_idx, s_idx, p_prob, s_prob,
                    writer
                )
    return out_csv

def process_predictions_parallel(
    results_list,
    hdf5_file,
    p_threshold,
    s_threshold,
    partial_prefix
):
    """
    多进程 => 分块 => pool.map => 每个子进程写 partial csv
    """
    import math
    from functools import partial

    n_proc= min(8, mp.cpu_count())
    chunk_size= math.ceil(len(results_list)/n_proc)
    chunks= [results_list[i:i+chunk_size] for i in range(0,len(results_list), chunk_size)]

    with mp.Pool(n_proc) as pool:
        func= partial(
            process_sublist,
            hdf5_file= hdf5_file,
            p_threshold= p_threshold,
            s_threshold= s_threshold,
            partial_prefix= partial_prefix
        )
        partials= pool.map(func, chunks)

    return partials

def final_deduplicate(partial_csv_list, final_csv):
    """
    合并 partial csv => 2秒全局去重
    """
    df_list=[]
    for csvf in partial_csv_list:
        if os.path.exists(csvf):
            df_list.append(pd.read_csv(csvf))
    if len(df_list)==0:
        pd.DataFrame().to_csv(final_csv, index=False)
        return 0

    big_df= pd.concat(df_list, ignore_index=True)
    big_df['event_start_time']= pd.to_datetime(big_df['event_start_time'], errors='coerce')
    big_df.sort_values(by='event_start_time', inplace=True)

    dedup=[]
    last_t= None
    for _, row in big_df.iterrows():
        t= row['event_start_time']
        if pd.isnull(t):
            dedup.append(row)
            continue
        if last_t is None:
            dedup.append(row)
            last_t= t
        else:
            delta= (t-last_t).total_seconds()
            if delta>=2:
                dedup.append(row)
                last_t= t
            else:
                # 重复 =>丢弃
                pass
    final_df= pd.DataFrame(dedup)
    final_df.to_csv(final_csv, index=False)
    return len(final_df)


###################################
# 4) 主函数
###################################
def phasenet_prediction_parallel(
    input_csv,
    input_hdf5,
    store_dir,
    p_threshold=0.3,
    s_threshold=0.3,
    sample_ratio=0.1,
    batch_size=64,
    device=None,
    random_seed=49
):
    start_time= time.time()
    device= device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # 读取 CSV + 随机子集
    df= pd.read_csv(input_csv)
    trace_list= df["trace_name"].tolist()
    random.seed(random_seed)
    sample_size= int(len(trace_list)* sample_ratio)
    trace_list= random.sample(trace_list, sample_size)
    print(f"[INFO] total {len(df)} => sampled {len(trace_list)}")

    # Dataset + Loader
    ds= SeismicDataset(input_hdf5, trace_list, "std")
    loader= DataLoader(ds, batch_size=batch_size, shuffle=False,
                       num_workers=min(mp.cpu_count(),32), pin_memory=True)

    # PhaseNet
    model= sbm.PhaseNet.from_pretrained("stead").to(device)
    model.eval()

    # GPU inference => p, s
    p_results=[]
    s_results=[]
    print("[INFO] PhaseNet inference ...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            data= batch['data'].float().to(device)
            out= model(data) # (B,2,N)
            #0是噪声数据
            p_results.append(out[:,1,:].cpu().numpy())
            s_results.append(out[:,2,:].cpu().numpy())
    p_all= np.concatenate(p_results, axis=0)
    s_all= np.concatenate(s_results, axis=0)

    # 构造 results_list => [(trace_name, p_array, s_array), ...]
    results_list=[]
    idx=0
    for tn in trace_list:
        results_list.append((tn, p_all[idx], s_all[idx]))
        idx+=1

    # 并行后处理
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    out_prefix= os.path.join(store_dir, f"phasenet_{random_seed}")
    print("[INFO] parallel post-processing ...")
    partials= process_predictions_parallel(
        results_list,
        input_hdf5,
        p_threshold,
        s_threshold,
        out_prefix
    )

    # 合并 + 2秒去重
    final_csv= out_prefix+ "_final.csv"
    kept= final_deduplicate(partials, final_csv)

    # 打印统计
    end_time= time.time()
    dt= end_time- start_time
    h= int(dt//3600)
    m= int((dt-3600*h)//60)
    s= dt- 3600*h- 60*m
    print(f"\n*** Done: {h}h {m}m {s:.2f}s, final picks= {kept}, CSV => {final_csv}")


###################################
# Demo
###################################
if __name__=="__main__":
    mp.set_start_method("spawn", force=True)
    input_csv= "/mnt/nn/Code/DATA/STEAD/STEAD.csv"
    input_hdf5= "/mnt/nn/Code/DATA/STEAD/STEAD.hdf5"
    store_dir= "/home/yuan/code/SeisMoE/data/STEAD_PhaseNet"

    phasenet_prediction_parallel(
        input_csv,
        input_hdf5,
        store_dir,
        p_threshold=0.3,
        s_threshold=0.3,
        sample_ratio=0.1, # 10%
        batch_size=200,
        device=None,
        random_seed=42
    )
