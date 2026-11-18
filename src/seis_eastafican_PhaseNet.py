#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import h5py
import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from torch.utils.data import Dataset, DataLoader
import seisbench.models as sbm
from tqdm import tqdm
from datetime import datetime, timedelta
import multiprocessing as mp
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import resample,butter,filtfilt

def detect_phases(p_arr, s_arr, p_thresh=0.3, s_thresh=0.3):
    """使用 scipy.signal.find_peaks 实现相位检测，
    逻辑：仅根据 p_arr 中幅值最高的峰确定 P，
    然后在 P 后面 6000 个采样点内，从 s_arr 中选择幅值最高的峰作为 S。
    """
    # 对 p_arr 和 s_arr 分别进行峰值检测
    p_peaks, p_props = find_peaks(p_arr, height=p_thresh, distance=100)
    s_peaks, s_props = find_peaks(s_arr, height=s_thresh, distance=100)

    # 选择 p_arr 中最高的峰作为 P
    if len(p_peaks) > 0:
        best_p_index = np.argmax(p_props['peak_heights'])
        best_p = p_peaks[best_p_index]
        p_conf = p_props['peak_heights'][best_p_index]
    else:
        best_p = None
        p_conf = None

    best_s = None
    s_conf = None

    if best_p is not None:
        # 筛选出在 P 后且距离 P 小于 6000 个采样点的所有 S 峰
        valid_s_indices = [i for i, s in enumerate(s_peaks) if s > best_p and (s - best_p) < 6000]
        if valid_s_indices:
            valid_s_peaks = s_peaks[valid_s_indices]
            valid_s_props = s_props['peak_heights'][valid_s_indices]
            best_s_idx = np.argmax(valid_s_props)
            best_s = valid_s_peaks[best_s_idx]
            s_conf = valid_s_props[best_s_idx]
    return best_p, best_s, p_conf, s_conf


class SeismicDataset(Dataset):
    def __init__(self, hdf5_path, trace_names, norm_mode='std', sample_rate=100.0):
        self.hdf5_path = hdf5_path
        # 预检查有效的trace_names
        with h5py.File(hdf5_path, 'r') as f:
            self.trace_names = [name for name in trace_names if name in f['data']]
        self.norm_mode = norm_mode
        self.sample_rate = sample_rate
        self.h5_file = None  # 延迟打开
        
        # 记录缺失的trace_name
        missing = set(trace_names) - set(self.trace_names)
        if missing:
            print(f"警告：共 {len(missing)} 个trace_name不存在于HDF5文件中")
            with open("missing_traces.txt", "w") as f:
                f.write("\n".join(sorted(missing)))

    def __len__(self):
        return len(self.trace_names)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')
        
        trace_name = self.trace_names[idx]
        
        try:
            # 读取数据并进行处理
            data = self.h5_file['data'][trace_name][()]  # shape: (3, T)
            data = self.bandpass_filter(data, lowcut=1.0, highcut=45.0, fs=self.sample_rate)
            data = self.normalize_data(data, mode=self.norm_mode)
            
            return {
                'trace_name': trace_name,
                'data': torch.from_numpy(data).float()  # 转换为(T, 3)形状
            }
        except Exception as e:
            print(f"处理 {trace_name} 时出错: {str(e)}")
            # 返回一个空数据占位符
            return self._get_dummy_item(trace_name)

    def _get_dummy_item(self, trace_name):
        """返回一个形状正确但值为0的虚拟数据"""
        dummy_data = np.zeros((6000, 3))  # 假设6000个时间点
        return {
            'trace_name': trace_name,
            'data': torch.from_numpy(dummy_data).float()
        }

    def normalize_data(self, data, mode='std'):
        data = data - np.mean(data, axis=1, keepdims=True)
        if mode == 'std':
            std_val = np.std(data, axis=1, keepdims=True)
            std_val[std_val < 1e-6] = 1.0
            data = data / std_val
        elif mode == 'max':
            max_val = np.max(np.abs(data), axis=1, keepdims=True)
            max_val[max_val < 1e-6] = 1.0
            data = data / max_val
        else:
            raise ValueError(f"不支持的标准化模式: {mode}")
        return data

    def bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')

        filtered = np.zeros_like(data)
        padlen = 3 * max(len(b), len(a))

        for ch in range(data.shape[0]):
            if data.shape[1] <= padlen:
                filtered[ch] = data[ch].copy()
            else:
                filtered[ch] = filtfilt(b, a, data[ch])
        return filtered

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()


def collate_fn(batch):
    """自定义批次处理函数，过滤无效数据"""
    # 过滤掉None值
    batch = [b for b in batch if b is not None]
    if not batch:
        # 如果整个批次都是无效数据，返回一个虚拟批次
        dummy_data = torch.zeros((1, 6000, 3))  # 假设6000个时间点
        return {
            'trace_name': ['dummy'],
            'data': dummy_data
        }
    return torch.utils.data.default_collate(batch)



def model_predict(model, data: torch.Tensor):
    with torch.no_grad():
        return model(data)

def main(input_csv, hdf5_path, output_dir, p_thresh=0.3, s_thresh=0.3, batch_size=256):
    # 读取 CSV 文件并随机采样 10% 的 trace_name
    random.seed(36)
    df = pd.read_csv(input_csv)
    trace_names = df['trace_name'].sample(frac=1).tolist()
    print(f"预测数据数量：{len(trace_names)}")
    
    loader = DataLoader(
        SeismicDataset(hdf5_path, trace_names),
        batch_size=batch_size,
        num_workers=mp.cpu_count(),
        pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载模型，此处使用 seisbench 提供的 EQTransformer 预训练模型
    model = sbm.PhaseNet.from_pretrained("original").to(device).eval()

    results = []
    for batch in tqdm(loader):
        data = batch['data'].to(device)
        # 进行模型推理，假设模型输出格式满足：output[1] 为 P 概率，output[2] 为 S 概率
        output = model_predict(model, data)
        p_prob = output[:,1,:].cpu()
        s_prob = output[:,2,:].cpu()
        
        # 对每个样本进行相位检测
        for i, (p_arr, s_arr) in enumerate(zip(p_prob, s_prob)):
            p_idx, s_idx, p_conf, s_conf = detect_phases(p_arr.numpy(), s_arr.numpy(), p_thresh, s_thresh)
            if p_idx is not None or s_idx is not None:
                results.append((batch['trace_name'][i], p_idx, s_idx, p_conf, s_conf))

    output_csv = os.path.join(output_dir, 'picks.csv')
    pd.DataFrame(results, columns=['trace_name', 'p_idx', 's_idx', 'p_conf', 's_conf'])\
      .to_csv(output_csv, index=False)
    
    print(f"****拾取到的事件数：{len(results)}")
    print(f"结果保存到：{output_csv}")



if __name__ == "__main__":
    # 设置多进程启动方式
    mp.set_start_method('spawn', force=True)
    
    main(
        input_csv=r'/home/cloudlab/Code_DL/DEEP_STEAD_PYTORCH/Pytorch_STEAD/DATA/东非裂谷数据/SE_ps_new_2.csv',
        hdf5_path=r'/home/cloudlab/Code_DL/DEEP_STEAD_PYTORCH/Pytorch_STEAD/DATA/东非裂谷数据/ES.hdf5',
        output_dir=r"/home/yuan/code/SeisMoE/data/EastAfican_Predict/EastAfican_PhaseNet",
        p_thresh=0.1,
        s_thresh=0.1,
    )