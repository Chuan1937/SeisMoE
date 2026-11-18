#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import sys
import h5py
import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime, timedelta
import multiprocessing as mp
from scipy.signal import resample,butter,filtfilt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# 将自定义模型文件所在目录加入路径
sys.path.append("/home/yuan/code/SeisMoE/src")
# 导入自定义模型文件，假设其中包含与 seisbench EQTransformer 接口相同的模型类
from model_moe_gpt_PS2_CPU import IMG1

def detect_phases(p_arr, s_arr, p_thresh=0.3, s_thresh=0.3):
    """使用 scipy.signal.find_peaks 实现相位检测"""
    p_peaks, p_props = find_peaks(p_arr, height=p_thresh, distance=100)
    s_peaks, s_props = find_peaks(s_arr, height=s_thresh, distance=100)

    best_p = p_peaks[p_props['peak_heights'].argmax()] if len(p_peaks) > 0 else None
    best_s = None

    if best_p is not None:
        for s in s_peaks:
            if 0 < (s - best_p) < 6000:  # 60秒窗口(100Hz采样)
                best_s = s
                break

    return best_p, best_s, p_props['peak_heights'].max() if best_p is not None else None, \
           s_props['peak_heights'][s_peaks.tolist().index(best_s)] if best_s is not None else None


class SeismicDataset(Dataset):
    def __init__(self, hdf5_path, trace_names, target_length=6000, norm_mode='std', sample_rate=100.0):
        self.hdf5_path = hdf5_path
        self.trace_names = trace_names
        self.target_length = target_length
        self.norm_mode = norm_mode
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.trace_names)

    def __getitem__(self, idx):
        trace_name = self.trace_names[idx]

        with h5py.File(self.hdf5_path, 'r') as f:
            if 'data' not in f:
                raise KeyError("HDF5文件中缺少'data'组")

            dataset = f['data'].get(trace_name)
            # 关键修改1：加载数据时直接转换为float32
            normalized = dataset[:].astype(np.float32).T  # shape: (3, n_samples)

        # 滤波：1-45Hz 带通滤波
        normalized = self.bandpass_filter(normalized, lowcut=1.0, highcut=45.0, fs=self.sample_rate)

        # 标准化
        normalized = self.normalize_data1(normalized,mode='std')

        return {
            'trace_name': trace_name,
            'data': torch.from_numpy(normalized.T).float()
        }

    def normalize_data1(self, data, mode='std'):
        data -= np.mean(data, axis=1, keepdims=True)

        if mode == 'std':
            std_val = np.std(data, axis=1, keepdims=True)
            std_val[std_val == 0] = 1.0
            data /= std_val
        elif mode == 'max':
            max_val = np.max(np.abs(data), axis=1, keepdims=True)
            max_val[max_val == 0] = 1.0
            data /= max_val
        else:
            raise ValueError(f"Unsupported norm_mode: {mode}")
        return data

    def normalize_data2(self, data, mode='max'):
        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert (max_data.shape[-1] == data.shape[-1])
            max_data[max_data == 0] = 1
            data /= max_data

        elif mode == 'std':
            std_data = np.std(data, axis=0, keepdims=True)
            assert (std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data

    def bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')

        filtered = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered[ch] = filtfilt(b, a, data[ch])
        return filtered


def model_predict(model, data: torch.Tensor):
    with torch.no_grad():
        return model(data)

def main(input_csv, hdf5_path, output_dir, p_thresh=0.3, s_thresh=0.3, batch_size=2048):
    # 读取 CSV 文件并采样10%的 trace_name
    df = pd.read_csv(input_csv)
    trace_names = df['trace_name'].sample(n=10000, random_state=36).tolist()
    print(f"预测数据数量：{len(trace_names)}")

    loader = DataLoader(
        SeismicDataset(hdf5_path, trace_names),
        batch_size=batch_size,
        num_workers=mp.cpu_count(),
        pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载自定义模型
    # 此处假设模型文件 model_moe_gpt_PS3_CPU.py 中定义的类名为 EQTransformer
    model = IMG1().to(device)
    model.load_state_dict(torch.load("/home/yuan/code/SeisMoE/models/best_model_CPU.pth",map_location=device))

    model.eval()

    results = []
    for batch in tqdm(loader):
        data = batch['data'].to(device)
        # 进行模型推理，假设返回格式与 seisbench 的 EQTransformer 一致，
        # 其中 output[1] 为 P 概率，output[2] 为 S 概率
        output,_ = model_predict(model, data)
        p_prob= output[0].cpu()
        s_prob= output[1].cpu()

        for i, (p_arr, s_arr) in enumerate(zip(p_prob, s_prob)):
            p_idx, s_idx, p_conf, s_conf = detect_phases(p_arr.numpy(), s_arr.numpy(), p_thresh, s_thresh)
            if p_idx is not None or s_idx is not None:
                results.append((batch['trace_name'][i], p_idx, s_idx, p_conf, s_conf))

    output_csv = os.path.join(output_dir, 'picks_moe2.csv')
    pd.DataFrame(results, columns=['trace_name', 'p_idx', 's_idx', 'p_conf', 's_conf']).to_csv(output_csv, index=False)

    print(f"****拾取到的事件数：{len(results)}")
    print(f"结果保存到：{output_csv}")

if __name__ == "__main__":
    main(
        input_csv=r"/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.csv",
        hdf5_path=r"/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.hdf5",
        output_dir=r"/home/yuan/code/SeisMoE/benchmark/diting_Moe",
        p_thresh=0.1,
        s_thresh=0.1
    )


