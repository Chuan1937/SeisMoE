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
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
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
        try:
            group = trace_name.split('$')[0].strip()
            index = int(trace_name.split('$')[1].split(',')[0].strip())
        except (IndexError, ValueError) as e:
            raise ValueError(f"无效的trace_name格式：'{trace_name}'") from e

        
        with h5py.File(self.hdf5_path, 'r') as f:
            if 'data' not in f:
                raise KeyError("HDF5文件中缺少'data'组")
            
            dataset = f['data'].get(group)
            if dataset is None:
                available_groups = list(f['data'].keys())
                raise KeyError(f"组'{group}'不存在，可用组：{available_groups}")
        
            
            try:
                raw_data = dataset[index, :, :].astype(np.float32)  # shape: (3, n_samples)
            except Exception as e:
                raise IndexError(f"读取数据失败（组：{group}，索引：{index}）") from e

        # 重采样到 target_length
        if raw_data.shape[1] != self.target_length:
            resampled = np.zeros((3, self.target_length), dtype=np.float32)
            for ch in range(3):
                resampled[ch] = resample(raw_data[ch], self.target_length)
        else:
            resampled = raw_data.astype(np.float32)



        # 滤波：1-45Hz 带通滤波
        filtered = self.bandpass_filter(resampled, lowcut=1.0, highcut=45.0, fs=self.sample_rate)

        # 标准化
        normalized = self.normalize_data1(filtered, mode='max')

        return {
            'trace_name': trace_name,
            'data': torch.from_numpy(normalized)  # shape: (T, 3)
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
        """
        对每个通道数据进行带通滤波。
        data: 形状为 (3, T)
        lowcut, highcut: 滤波器截止频率
        fs: 采样率
        """
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
    # 读取 CSV 文件并随机采样 10% 的 trace_name
    random.seed(36)
    df = pd.read_csv(input_csv)
    trace_names = df['trace_name'].sample(frac=0.12,random_state=36).tolist()
    print(f"预测数据数量：{len(trace_names)}")
    
    loader = DataLoader(
        SeismicDataset(hdf5_path, trace_names),
        batch_size=batch_size,
        num_workers=mp.cpu_count(),
        pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载模型，此处使用 seisbench 提供的 EQTransformer 预训练模型
    model = sbm.EQTransformer.from_pretrained("ethz").to(device).eval()

    results = []
    for batch in tqdm(loader):
        data = batch['data'].to(device)
        # 进行模型推理，假设模型输出格式满足：output[1] 为 P 概率，output[2] 为 S 概率
        output = model_predict(model, data)
        p_prob = output[1].cpu()
        s_prob = output[2].cpu()
        
        # 对每个样本进行相位检测
        for i, (p_arr, s_arr) in enumerate(zip(p_prob, s_prob)):
            p_idx, s_idx, p_conf, s_conf = detect_phases(p_arr.numpy(), s_arr.numpy(), p_thresh, s_thresh)
            if p_idx is not None or s_idx is not None:
                results.append((batch['trace_name'][i], p_idx, s_idx, p_conf, s_conf))

    output_csv = os.path.join(output_dir, 'picks_moe2.csv')
    pd.DataFrame(results, columns=['trace_name', 'p_idx', 's_idx', 'p_conf', 's_conf'])\
      .to_csv(output_csv, index=False)
    
    print(f"****拾取到的事件数：{len(results)}")
    print(f"结果保存到：{output_csv}")



if __name__ == "__main__":
    # 设置多进程启动方式
    mp.set_start_method('spawn', force=True)
    
    main(
        input_csv=r'/home/cloudlab/意大利数据集instancegm/Instance.csv',
        hdf5_path=r'/home/cloudlab/意大利数据集instancegm/Instance.hdf5',
        output_dir=r"/home/yuan/code/SeisMoE/data/Instance_Predict/Instance_EQTransformer_Predict",
        p_thresh=0.1,
        s_thresh=0.1,
    )