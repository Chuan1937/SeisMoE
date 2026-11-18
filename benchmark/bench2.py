#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""公平benchmark: 每个数据集内相同数据加载，测量总耗时+纯推理"""
import os, sys, time, h5py, csv, json
import numpy as np, pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks, butter, filtfilt, resample
from tqdm import tqdm

sys.path.insert(0, "/home/yuan/code/SeisMoE/src")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEV = torch.device('cuda')
MOE_W = "/home/yuan/code/SeisMoE/models/best_model_CPU.pth"
OUT = "/home/yuan/code/SeisMoE/benchmark"


# ============================================================
# Dataset 工厂：每个数据集一种预处理，两个shape变体
# ============================================================

def _bandpass(data, low=1.0, high=45.0, fs=100.0, order=4):
    """跳过太短的波形"""
    padlen = 3 * (order * 2)
    if data.shape[1] <= padlen:
        return data  # 太短，跳过滤波
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    out = np.zeros_like(data)
    for ch in range(data.shape[0]):
        out[ch] = filtfilt(b, a, data[ch])
    return out

def _normalize(data, axis=0):
    data = data - data.mean(axis=axis, keepdims=True)
    s = data.std(axis=axis, keepdims=True); s[s == 0] = 1.0
    return data / s

def _resample_6000(data):
    if data.shape[1] == 6000:
        return data
    out = np.zeros((3, 6000), dtype=np.float32)
    for ch in range(3):
        out[ch] = resample(data[ch], 6000)
    return out


# --- Txed: group$index, 无滤波 ---
class DS_Txed_3T(Dataset):
    def __init__(self, h5, names): self.h5, self.names = h5, names
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        tn = self.names[i]; g = tn.split('$')[0].strip(); idx = int(tn.split('$')[1].split(',')[0].strip())
        d = self.h5['data'][g][idx, :, :].astype(np.float32)
        d = _normalize(d, axis=0)
        return torch.from_numpy(d)  # (3,T)
class DS_Txed_T3(DS_Txed_3T):
    def __getitem__(self, i): return super().__getitem__(i).T  # (T,3)

# --- DiTing: direct key, bandpass ---
class DS_DiTing_3T(Dataset):
    def __init__(self, h5, names): self.h5, self.names = h5, names
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        d = self.h5['data'][self.names[i]][()].astype(np.float32).T  # HDF5存(T,3)→转(3,T)
        d = _bandpass(d); d = _normalize(d, axis=0)
        return torch.from_numpy(d)
class DS_DiTing_T3(DS_DiTing_3T):
    def __getitem__(self, i): return super().__getitem__(i).T

# --- Instance: group$index, resample + bandpass ---
class DS_Instance_3T(Dataset):
    def __init__(self, h5, names): self.h5, self.names = h5, names
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        tn = self.names[i]; g = tn.split('$')[0].strip(); idx = int(tn.split('$')[1].split(',')[0].strip())
        d = self.h5['data'][g][idx, :, :].astype(np.float32)
        d = _resample_6000(d); d = _bandpass(d); d = _normalize(d, axis=0)
        return torch.from_numpy(d)
class DS_Instance_T3(DS_Instance_3T):
    def __getitem__(self, i): return super().__getitem__(i).T

# --- STEAD: direct key, bandpass ---
class DS_STEAD_3T(Dataset):
    def __init__(self, h5, names): self.h5, self.names = h5, names
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        d = self.h5['data'][self.names[i]][()].astype(np.float32).T  # (T,3)→(3,T)
        d = _normalize(d, axis=0)  # 无bandpass，提速
        return torch.from_numpy(d)
class DS_STEAD_T3(DS_STEAD_3T):
    def __getitem__(self, i): return super().__getitem__(i).T


# ============================================================
# 运行单个模型测试
# ============================================================
def run_one(label, h5_path, csv_path, ds_cls, model_type, n_samples, batch_size):
    print(f"\n{'='*55}\n  {label} | {n_samples}条 | bs={batch_size}")
    df = pd.read_csv(csv_path)
    names = df['trace_name'].sample(n=n_samples, random_state=36).tolist()

    h5 = h5py.File(h5_path, 'r')
    ds = ds_cls(h5, names)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=10,
                        pin_memory=True, prefetch_factor=3)

    # 加载模型
    if model_type == 'eq':
        import seisbench.models as sbm
        model = sbm.EQTransformer.from_pretrained("original").to(DEV).eval()
    elif model_type == 'pn':
        import seisbench.models as sbm
        model = sbm.PhaseNet.from_pretrained("original").to(DEV).eval()
    else:  # moe
        from model_moe_gpt_PS3_CPU import IMG1
        model = IMG1()
        model.load_state_dict(torch.load(MOE_W, map_location='cpu'))
        model = model.to(DEV).eval()

    # warmup
    for b in loader:
        data = b.to(DEV, non_blocking=True)
        if model_type == 'moe': model(data)
        else: model(data)
        break

    # 计时
    torch.cuda.synchronize()
    t0 = time.time()
    infer = 0.0
    with torch.no_grad():
        for b in tqdm(loader, desc=f"  {label.split('/')[1]}"):
            data = b.to(DEV, non_blocking=True)
            torch.cuda.synchronize()
            t1 = time.time()
            if model_type == 'moe':
                model(data)
            else:
                model(data)
            torch.cuda.synchronize()
            infer += time.time() - t1
    total = time.time() - t0

    del model, loader, ds
    h5.close()
    torch.cuda.empty_cache()

    print(f"  => 总:{total:.1f}s  推理:{infer:.1f}s")
    return total, infer


# ============================================================
# 主流程
# ============================================================
def main():
    all_results = {}

    configs = [
        # --- Txed 1万 ---
        ("Txed/EQTrans",  "/mnt/d/AI_Seismic_Data/txed/Texd.csv",
         "/mnt/d/AI_Seismic_Data/txed/Texd.hdf5", DS_Txed_3T, 'eq', 10000, 1024),
        ("Txed/PhaseNet","/mnt/d/AI_Seismic_Data/txed/Texd.csv",
         "/mnt/d/AI_Seismic_Data/txed/Texd.hdf5", DS_Txed_3T, 'pn', 10000, 1024),
        ("Txed/MoE",     "/mnt/d/AI_Seismic_Data/txed/Texd.csv",
         "/mnt/d/AI_Seismic_Data/txed/Texd.hdf5", DS_Txed_T3, 'moe', 10000, 1024),

        # --- DiTing 1万 ---
        ("DiTing/EQTrans","/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.csv",
         "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.hdf5",
         DS_DiTing_3T, 'eq', 10000, 1024),
        ("DiTing/PhaseNet","/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.csv",
         "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.hdf5",
         DS_DiTing_3T, 'pn', 10000, 1024),
        ("DiTing/MoE","/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.csv",
         "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.hdf5",
         DS_DiTing_T3, 'moe', 10000, 1024),

        # --- Instance 10万 ---
        ("Instance/EQTrans","/mnt/d/AI_Seismic_Data/Instance/INSTANCE.csv",
         "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.hdf5",
         DS_Instance_3T, 'eq', 100000, 1024),
        ("Instance/PhaseNet","/mnt/d/AI_Seismic_Data/Instance/INSTANCE.csv",
         "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.hdf5",
         DS_Instance_3T, 'pn', 100000, 1024),
        ("Instance/MoE","/mnt/d/AI_Seismic_Data/Instance/INSTANCE.csv",
         "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.hdf5",
         DS_Instance_T3, 'moe', 100000, 1024),

        # --- STEAD 10万 ---
        ("STEAD/EQTrans","/mnt/d/AI_Seismic_Data/STEAD/STEAD.csv",
         "/mnt/d/AI_Seismic_Data/STEAD/STEAD.hdf5",
         DS_STEAD_3T, 'eq', 100000, 1024),
        ("STEAD/PhaseNet","/mnt/d/AI_Seismic_Data/STEAD/STEAD.csv",
         "/mnt/d/AI_Seismic_Data/STEAD/STEAD.hdf5",
         DS_STEAD_3T, 'pn', 100000, 1024),
        ("STEAD/MoE","/mnt/d/AI_Seismic_Data/STEAD/STEAD.csv",
         "/mnt/d/AI_Seismic_Data/STEAD/STEAD.hdf5",
         DS_STEAD_T3, 'moe', 100000, 1024),
    ]

    # 跳过已完成的
    skip = {'Txed', 'Instance','DiTing'}  
    
    for label, csv_path, h5_path, ds_cls, model_type, n_samples, bs in configs:
        ds_name = label.split('/')[0]
        if ds_name in skip:
            print(f"\n  SKIP {label} (已完成)")
            continue
        try:
            total, infer = run_one(label, h5_path, csv_path, ds_cls, model_type, n_samples, bs)
            all_results[label] = (total, infer)
        except Exception as e:
            print(f"  *** FAIL: {e}")
            all_results[label] = (None, None)

    # 保存CSV
    with open(f'{OUT}/all_times.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['数据集','模型','总耗时(s)','纯推理(s)'])
        for label, (t, i) in all_results.items():
            ds, m = label.split('/')
            w.writerow([ds, m, f'{t:.1f}' if t else 'FAIL',
                        f'{i:.1f}' if i else 'FAIL'])

    # 打印汇总
    print("\n" + "=" * 55)
    print("汇总:")
    for ds in ['Txed','DiTing','Instance','STEAD']:
        print(f"  {ds}:")
        for m in ['EQTrans','PhaseNet','MoE']:
            t, i = all_results.get(f'{ds}/{m}', (None, None))
            print(f"    {m}: 总={t:.1f}s 推理={i:.1f}s" if t else f"    {m}: N/A")

    print(f"\nCSV: {OUT}/all_times.csv")


if __name__ == "__main__":
    main()
