#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化版本 - seis_Texd_phasenet
针对64GB内存和8GB显存优化
"""

import os
import random
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import multiprocessing as mp
from scipy.signal import find_peaks
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import seisbench.models as sbm


class SeismicDataset(Dataset):
    """优化的数据集类 - 保持HDF5文件句柄打开"""
    
    def __init__(self, hdf5_path, trace_names, norm_mode='std'):
        self.hdf5_path = hdf5_path
        self.trace_names = trace_names
        self.norm_mode = norm_mode
                
    def _open_hdf5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')
    
    def __len__(self):
        return len(self.trace_names)
    
    def __getitem__(self, idx):
        self._open_hdf5()
        
        trace_name = self.trace_names[idx]
        try:
            group = trace_name.split('$')[0].strip()
            index = int(trace_name.split('$')[1].split(',')[0].strip())
        except (IndexError, ValueError) as e:
            raise ValueError(f"无效的trace_name格式：{trace_name}") from e
        
        try:
            dataset = self.h5_file['data'].get(group)
            if dataset is None:
                raise KeyError(f"组'{group}'不存在")
            data = dataset[index, :, :].astype(np.float32)
        except Exception as e:
            raise IndexError(f"读取数据失败") from e
        
        # 标准化
        if self.norm_mode == 'std':
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            std[std == 0] = 1.0
            data = (data - mean) / std
        elif self.norm_mode == 'max':
            max_val = np.max(np.abs(data), axis=1, keepdims=True)
            max_val[max_val == 0] = 1.0
            data = data / max_val
        
        return {
            'trace_name': trace_name,
            'data': torch.from_numpy(data)
        }
    
    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()


def detect_phases_batch(p_probs, s_probs, p_thresh=0.3, s_thresh=0.3):
    """批量相位检测"""
    batch_size = p_probs.shape[0]
    results = []
    
    for i in range(batch_size):
        p_arr = p_probs[i].numpy()
        s_arr = s_probs[i].numpy()
        
        p_peaks, p_props = find_peaks(p_arr, height=p_thresh, distance=100)
        if len(p_peaks) == 0:
            continue
        
        best_p_idx = np.argmax(p_props['peak_heights'])
        best_p = p_peaks[best_p_idx]
        p_conf = p_props['peak_heights'][best_p_idx]
        
        s_peaks, s_props = find_peaks(s_arr, height=s_thresh, distance=100)
        best_s = None
        s_conf = None
        
        if len(s_peaks) > 0:
            valid_mask = (s_peaks > best_p) & (s_peaks - best_p < 6000)
            valid_s_peaks = s_peaks[valid_mask]
            if len(valid_s_peaks) > 0:
                valid_s_heights = s_props['peak_heights'][np.where(valid_mask)[0]]
                best_s_idx = np.argmax(valid_s_heights)
                best_s = valid_s_peaks[best_s_idx]
                s_conf = valid_s_heights[best_s_idx]
        
        if best_p is not None or best_s is not None:
            results.append((best_p, best_s, p_conf, s_conf))
    
    return results


def main(input_csv, hdf5_path, output_dir, p_thresh=0.3, s_thresh=0.3, batch_size=2048):
    random.seed(36)
    df = pd.read_csv(input_csv)
    trace_names = df['trace_name'].sample(n=10000, random_state=36).tolist()
    print(f"预测数据数量：{len(trace_names)}")
    
    dataset = SeismicDataset(hdf5_path, trace_names)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = sbm.PhaseNet.from_pretrained("original").to(device).eval()
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="推理中"):
            data = batch['data'].to(device, non_blocking=True)
            
            output = model(data)
            
            p_prob = output[:, 1, :].cpu()
            s_prob = output[:, 2, :].cpu()
            
            batch_results = detect_phases_batch(p_prob, s_prob, p_thresh, s_thresh)
            
            for j, (p_idx, s_idx, p_conf, s_conf) in enumerate(batch_results):
                results.append((batch['trace_name'][j], p_idx, s_idx, p_conf, s_conf))
    
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'picks_moe2.csv')
    pd.DataFrame(results, columns=['trace_name', 'p_idx', 's_idx', 'p_conf', 's_conf']).to_csv(output_csv, index=False)
    
    print(f"拾取到的事件数：{len(results)}")
    print(f"结果保存到：{output_csv}")


if __name__ == "__main__":
    start = time.time()
    
    main(
        input_csv="/mnt/d/AI_Seismic_Data/txed/Texd.csv",
        hdf5_path="/mnt/d/AI_Seismic_Data/txed/Texd.hdf5",
        output_dir="/home/yuan/code/SeisMoE/benchmark/Texd_phasenet",
        p_thresh=0.1,
        s_thresh=0.1,
        batch_size=2048,
    )
    
    end = time.time()
    elapsed = end - start
    
    time_file = os.path.join("/home/yuan/code/SeisMoE/benchmark/Texd_phasenet", "time.txt")
    with open(time_file, 'w') as f:
        f.write(str(elapsed))
    
    print(f"运行时间: {elapsed:.2f}秒")
