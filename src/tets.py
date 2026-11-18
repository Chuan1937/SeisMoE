#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PhaseNet 预测与后处理流程 —— 利用 seisbench.annotations 标准化输出
"""

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

# 导入 seisbench 的 annotations 模块（请确保您的 seisbench 版本支持此接口）
from seisbench.annotations import Annotation, AnnotationCollection


###################################
# 1) 工具函数
###################################
def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):
    """
    峰值检测函数：检测一维数组 x 中的峰值，支持设置最小峰值高度、最小间隔等参数。
    参数 valley=True 时，检测谷值（即负值转正后检测峰值）。
    """
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    dx = x[1:] - x[:-1]

    # 处理 NaN
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

    # 去除 NaN 邻域的峰值
    if ind.size and indnan.size:
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # 去除边界值
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # 最小峰值高度限制
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # 阈值限制
    if ind.size and threshold > 0:
        dx_min = np.min(np.vstack([x[ind] - x[ind-1], x[ind] - x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx_min < threshold)[0])
    # 保证最小间隔
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
    对于一个 trace 内的预测结果：
      - 若存在 P 波，则只保留概率最高的 P，并在其后 60 秒内寻找首个符合条件的 S 波；
      - 若无 P 波，则单独选取概率最高的 S 波。
    返回形式：
         { best_p_idx: (best_s_idx or None, best_p_prob, best_s_prob or None) }
    """
    matches = {}
    # 处理 P 波拾取
    p_inds = _detect_peaks(p_array, mph=P_threshold, mpd=1)
    if len(p_inds) > 0:
        best_p_idx = p_inds[p_array[p_inds].argmax()]
        best_p_prob = p_array[best_p_idx]
        # 在 P 波后 60s 内寻找 S 波
        s_inds = _detect_peaks(s_array, mph=S_threshold, mpd=1)
        best_s_idx = None
        best_s_prob = None
        p_time_sec = best_p_idx / 100.0  # 假设采样率为 100 Hz
        for s_idx in s_inds:
            s_time_sec = s_idx / 100.0
            if 0 < (s_time_sec - p_time_sec) < 60:
                best_s_idx = s_idx
                best_s_prob = s_array[s_idx]
                break
        matches[best_p_idx] = (best_s_idx, best_p_prob, best_s_prob)
    else:
        # 若无 P 波，则处理 S 波拾取
        s_inds = _detect_peaks(s_array, mph=S_threshold, mpd=1)
        if len(s_inds) > 0:
            best_s_idx = s_inds[s_array[s_inds].argmax()]
            best_s_prob = s_array[best_s_idx]
            matches[None] = (best_s_idx, None, best_s_prob)
    return matches


def _get_snr(data, pick_index, window=200):
    """
    计算信噪比（SNR）：利用 pick_index 周围 window 长度内的噪声和信号计算 SNR。
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
        except Exception as e:
            pass
    return snr


###################################
# 2) Annotation 构造函数
###################################
def create_annotation(dataset_h5, trace_name, p_idx, s_idx, p_prob, s_prob, input_csv):
    """
    根据 HDF5 数据和 CSV 元数据，构造一个 Annotation 对象。
    采样率假设为 100 Hz。
    """
    # 读取元数据（假设 CSV 中每个 trace 只有一行记录）
    df = pd.read_csv(input_csv)
    meta_df = df[df['trace_name'] == trace_name]
    if meta_df.empty:
        raise ValueError(f"元数据中未找到 trace_name: {trace_name}")
    meta = meta_df.iloc[0]
    station_lat = meta['station_latitude_deg']
    station_lon = meta['station_longitude_deg']
    station_elv = meta['station_elevation_m']
    start_time = pd.to_datetime(meta['source_origin_time'])

    # 根据采样率 100 Hz 计算 P/S 到时
    p_time = start_time + timedelta(seconds=p_idx / 60.0) if p_idx is not None else None
    s_time = start_time + timedelta(seconds=s_idx / 60.0) if s_idx is not None else None

    # 从 HDF5 中读取数据，计算 SNR
    dset = dataset_h5['data'][trace_name]
    data_array = np.array(dset, dtype=np.float32)
    p_snr = _get_snr(data_array[:, 0], p_idx, window=200) if p_idx is not None else None
    s_snr = _get_snr(data_array[:, 1], s_idx, window=200) if s_idx is not None else None

    # 构造 Annotation 对象，picks 存储各相信息
    annotation = Annotation(
        trace_id=trace_name,
        station_lat=station_lat,
        station_lon=station_lon,
        station_elv=station_elv,
        picks={
            "P": {
                "time": p_time.strftime('%Y-%m-%dT%H:%M:%S.%f') if p_time is not None else None,
                "probability": round(p_prob, 3) if p_prob is not None else None,
                "snr": p_snr
            },
            "S": {
                "time": s_time.strftime('%Y-%m-%dT%H:%M:%S.%f') if s_time is not None else None,
                "probability": round(s_prob, 3) if s_prob is not None else None,
                "snr": s_snr
            }
        }
    )
    return annotation


###################################
# 3) 数据集与并行后处理
###################################
class SeismicDataset(Dataset):
    """
    读取 HDF5 文件中数据，不在 __init__ 中常驻打开文件，
    以支持多进程 DataLoader，每次 __getitem__ 时临时打开文件。
    """
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
        # 数据归一化
        if self.normalization_mode == 'std':
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-10)
        elif self.normalization_mode == 'max':
            data = data / (np.max(np.abs(data), axis=0) + 1e-10)
        return {"trace_name": trace_name, "data": torch.from_numpy(data)}


def process_sublist(sublist, hdf5_file, p_threshold, s_threshold, partial_prefix, input_csv):
    """
    每个子进程处理一个子列表，并利用 tqdm 显示进度。
    对每个 trace，根据预测结果构造 Annotation 对象，并保存到部分 CSV 文件中。
    """
    annotations = AnnotationCollection()
    with h5py.File(hdf5_file, 'r') as h5:
        for (trace_name, p_arr, s_arr) in tqdm(sublist, desc=f"Worker {os.getpid()}", total=len(sublist)):
            matches = phasenet_picker(p_threshold, s_threshold, p_arr, s_arr)
            for p_idx, (s_idx, p_prob, s_prob) in matches.items():
                try:
                    ann = create_annotation(h5, trace_name, p_idx, s_idx, p_prob, s_prob, input_csv)
                    annotations.add(ann)
                except Exception as e:
                    print(f"Error processing {trace_name}: {e}", file=sys.stderr)
    partial_file = partial_prefix + f".pid_{os.getpid()}_annotations.csv"
    annotations.to_csv(partial_file)
    return partial_file


def process_predictions_parallel(results_list, hdf5_file, p_threshold, s_threshold, partial_prefix, input_csv):
    """
    多进程处理：将 results_list 分块后，调用 process_sublist 生成多个部分 Annotation CSV 文件。
    """
    import math
    from functools import partial

    n_proc = min(8, mp.cpu_count())
    chunk_size = math.ceil(len(results_list) / n_proc)
    chunks = [results_list[i:i+chunk_size] for i in range(0, len(results_list), chunk_size)]

    with mp.Pool(n_proc) as pool:
        func = partial(
            process_sublist,
            hdf5_file=hdf5_file,
            p_threshold=p_threshold,
            s_threshold=s_threshold,
            partial_prefix=partial_prefix,
            input_csv=input_csv
        )
        partial_files = pool.map(func, chunks)
    return partial_files


def final_deduplicate(partial_csv_list, final_csv):
    """
    合并所有部分 Annotation CSV 文件，并对事件的 P 到时 (p_time) 做全局 2 秒去重，
    最后保存最终的 Annotation CSV 文件。
    """
    df_list = []
    for csvf in partial_csv_list:
        if os.path.exists(csvf):
            df_list.append(pd.read_csv(csvf))
    if len(df_list) == 0:
        pd.DataFrame().to_csv(final_csv, index=False)
        return 0

    big_df = pd.concat(df_list, ignore_index=True)
    # 假设 CSV 中存储的 P 到时字段名为 "P_time"
    # 请根据实际 Annotation CSV 文件中的字段名称进行修改
    big_df['P_time'] = pd.to_datetime(big_df['P_time'], errors='coerce')
    big_df.sort_values(by='P_time', inplace=True)

    dedup = []
    last_t = None
    for _, row in big_df.iterrows():
        t = row['P_time']
        if pd.isnull(t):
            dedup.append(row)
            continue
        if last_t is None:
            dedup.append(row)
            last_t = t
        else:
            delta = (t - last_t).total_seconds()
            if delta >= 2:
                dedup.append(row)
                last_t = t
    final_df = pd.DataFrame(dedup)
    final_df.to_csv(final_csv, index=False)
    return len(final_df)


###################################
# 4) 主流程函数：PhaseNet 预测及后处理
###################################
def phasenet_prediction_parallel(input_csv, input_hdf5, store_dir,
                                   p_threshold=0.3, s_threshold=0.3,
                                   sample_ratio=0.1, batch_size=64,
                                   device=None, random_seed=49):
    start_time = time.time()
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # 读取 CSV 元数据，并随机抽样部分 trace 进行预测
    df = pd.read_csv(input_csv)
    trace_list = df["trace_name"].tolist()
    random.seed(random_seed)
    sample_size = int(len(trace_list) * sample_ratio)
    trace_list = random.sample(trace_list, sample_size)
    print(f"[INFO] 总记录数: {len(df)}，抽样数: {len(trace_list)}")

    # 创建 Dataset 与 DataLoader
    ds = SeismicDataset(input_hdf5, trace_list, normalization_mode="std")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=mp.cpu_count(), pin_memory=True)

    # 加载 PhaseNet 模型（使用 seisbench.models 中预训练模型）
    model = sbm.PhaseNet.from_pretrained("stead").to(device)
    model.eval()

    # GPU 推理：存储所有 trace 的预测结果（N, P, S）
    n_results, p_results, s_results = [], [], []
    print("[INFO] PhaseNet 推理中 ...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            data = batch['data'].float().to(device)
            out = model(data)  # 输出形状为 (B, 2, N)，其中各通道分别代表 N, P, S
            n_results.append(out[:, 0, :].cpu().numpy())
            p_results.append(out[:, 1, :].cpu().numpy())
            s_results.append(out[:, 2, :].cpu().numpy())

    n_all = np.concatenate(n_results, axis=0)
    p_all = np.concatenate(p_results, axis=0)
    s_all = np.concatenate(s_results, axis=0)

    # 保存中间结果，便于后续分析
    np.savez(os.path.join(store_dir, "phasenet.npz"),
             T=trace_list,
             N=n_all,
             P=p_all,
             S=s_all)

    # 构造结果列表，每个元素为 (trace_name, p_array, s_array)
    results_list = []
    for idx, tn in enumerate(trace_list):
        results_list.append((tn, p_all[idx], s_all[idx]))

    # 并行后处理：每个子进程生成部分 Annotation CSV 文件
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    partial_prefix = os.path.join(store_dir, f"phasenet_{random_seed}")
    print("[INFO] 并行后处理 ...")
    partial_files = process_predictions_parallel(
        results_list,
        input_hdf5,
        p_threshold,
        s_threshold,
        partial_prefix,
        input_csv
    )

    # 合并部分文件，并对事件进行 2 秒去重
    final_csv = partial_prefix + "_final_annotations.csv"
    kept = final_deduplicate(partial_files, final_csv)

    end_time = time.time()
    dt = end_time - start_time
    h = int(dt // 3600)
    m = int((dt - 3600 * h) // 60)
    s_time_sec = dt - 3600 * h - 60 * m
    print(f"\n*** 完成：{h}h {m}m {s_time_sec:.2f}s, 最终标注数: {kept}, 输出文件: {final_csv}")


###################################
# 5) Demo 主程序入口
###################################
if __name__ == "__main__":
    # 为多进程设置 spawn 方法
    mp.set_start_method("spawn", force=True)
    
    # 输入文件路径（请根据实际情况修改）
    input_csv = "DATA/metadata_no_noise.csv"
    input_hdf5 = "DATA/waveforms_all_nonoise.hdf5"
    store_dir = "/home/yuan/code/SeisMoE/data/test"

    phasenet_prediction_parallel(
        input_csv=input_csv,
        input_hdf5=input_hdf5,
        store_dir=store_dir,
        p_threshold=0.1,
        s_threshold=0.1,
        sample_ratio=0.0001,  # 可根据需要调整抽样比例
        batch_size=200,
        device=None,
        random_seed=36
    )
