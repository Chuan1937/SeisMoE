#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一benchmark: 12个测试在一个进程中顺序执行
- 每个test保持原始脚本的预处理逻辑
- HDF5文件句柄缓存
- GPU同步计时
- 进度条可见
"""
import os, sys, time, random, h5py, json, re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks, butter, filtfilt, resample
from tqdm import tqdm

sys.path.insert(0, "/home/yuan/code/SeisMoE/src")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# === Dataset classes matching originals ===

class SeismicDataset_Texd(Dataset):
    """Txed: group$index, normalize_data2 std axis=0, no bandpass"""
    def __init__(self, h5, names): self.h5, self.names = h5, names
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        tn = self.names[i]
        g = tn.split('$')[0].strip()
        idx = int(tn.split('$')[1].split(',')[0].strip())
        data = self.h5['data'][g][idx,:,:].astype(np.float32)  # (3,T)
        data -= data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True); std[std==0]=1.0
        data /= std
        return torch.from_numpy(data)

class SeismicDataset_Texd_MoE(Dataset):
    """Txed MoE: group$index, normalize_data1 std axis=1, no bandpass, (T,3)"""
    def __init__(self, h5, names): self.h5, self.names = h5, names
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        tn = self.names[i]
        g = tn.split('$')[0].strip()
        idx = int(tn.split('$')[1].split(',')[0].strip())
        data = self.h5['data'][g][idx,:,:].astype(np.float32)  # (3,T)
        data -= data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True); std[std==0]=1.0
        data /= std
        return torch.from_numpy(data.T)  # (T,3)

class SeismicDataset_Instance(Dataset):
    """Instance: group$index, resample 6000, bandpass 1-45Hz, normalize_data1 max axis=1"""
    def __init__(self, h5, names): self.h5, self.names = h5, names
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        tn = self.names[i]
        g = tn.split('$')[0].strip()
        idx = int(tn.split('$')[1].split(',')[0].strip())
        data = self.h5['data'][g][idx,:,:].astype(np.float32)  # (3,T)
        # resample to 6000
        if data.shape[1] != 6000:
            d2 = np.zeros((3,6000), dtype=np.float32)
            for ch in range(3): d2[ch] = resample(data[ch], 6000)
            data = d2
        # bandpass 1-45Hz
        data = _bandpass(data)
        # normalize_data1 max axis=1
        data -= data.mean(axis=1, keepdims=True)
        mx = np.max(np.abs(data), axis=1, keepdims=True); mx[mx==0]=1.0
        data /= mx
        return torch.from_numpy(data)

class SeismicDataset_Instance_MoE(Dataset):
    """Instance MoE: same as Instance but (T,3)"""
    def __init__(self, h5, names): self.h5, self.names = h5, names
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        tn = self.names[i]
        g = tn.split('$')[0].strip()
        idx = int(tn.split('$')[1].split(',')[0].strip())
        data = self.h5['data'][g][idx,:,:].astype(np.float32)
        if data.shape[1] != 6000:
            d2 = np.zeros((3,6000), dtype=np.float32)
            for ch in range(3): d2[ch] = resample(data[ch], 6000)
            data = d2
        data = _bandpass(data)
        data -= data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True); std[std==0]=1.0
        data /= std
        return torch.from_numpy(data.T)

class SeismicDataset_DiTing(Dataset):
    """DiTing: direct key, bandpass 1-45Hz, normalize_data2 std axis=0, (T,3)"""
    def __init__(self, h5, names): self.h5, self.names = h5, names
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        data = self.h5['data'][self.names[i]][()].astype(np.float32)  # (3,T)
        data = _bandpass(data)
        data -= data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True); std[std==0]=1.0
        data /= std
        return torch.from_numpy(data.T)

class SeismicDataset_DiTing_MoE(Dataset):
    """DiTing MoE: direct key, bandpass 1-45Hz, normalize_data1 std axis=1, (T,3)"""
    def __init__(self, h5, names): self.h5, self.names = h5, names
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        data = self.h5['data'][self.names[i]][()].astype(np.float32)
        data = _bandpass(data)
        data -= data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True); std[std==0]=1.0
        data /= std
        return torch.from_numpy(data.T)

class SeismicDataset_STEAD(Dataset):
    """STEAD: direct key, bandpass 1-45Hz, normalize_data max axis=0, (T,3)"""
    def __init__(self, h5, names): self.h5, self.names = h5, names
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        data = self.h5['data'][self.names[i]][()].astype(np.float32)
        data = _bandpass(data)
        data -= data.mean(axis=0, keepdims=True)
        mx = np.max(data, axis=0, keepdims=True); mx[mx==0]=1.0
        data /= mx
        return torch.from_numpy(data.T)

class SeismicDataset_STEAD_MoE(Dataset):
    """STEAD MoE: direct key, bandpass 1-40Hz, normalize_data std axis=1, (T,3)"""
    def __init__(self, h5, names): self.h5, self.names = h5, names
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        data = self.h5['data'][self.names[i]][()].astype(np.float32)
        data = _bandpass(data, highcut=40.0)
        data -= data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True); std[std==0]=1.0
        data /= std
        return torch.from_numpy(data.T)

# 带通滤波(缓存系数)
_bandpass_cache = {}
def _bandpass(data, lowcut=1.0, highcut=45.0, fs=100.0, order=4):
    key = (lowcut, highcut)
    if key not in _bandpass_cache:
        nyq = 0.5 * fs
        b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
        _bandpass_cache[key] = (b, a)
    b, a = _bandpass_cache[key]
    out = np.zeros_like(data)
    for ch in range(data.shape[0]):
        out[ch] = filtfilt(b, a, data[ch])
    return out


def detect_phases_batch(pp, sp, pt=0.3, st=0.3):
    res = []
    for i in range(pp.shape[0]):
        pa, sa = pp[i].numpy(), sp[i].numpy()
        pk, pp_ = find_peaks(pa, height=pt, distance=100)
        if not len(pk): continue
        bi = np.argmax(pp_['peak_heights'])
        bp, pc = pk[bi], pp_['peak_heights'][bi]
        sk, sp_ = find_peaks(sa, height=st, distance=100)
        bs, sc = None, None
        if len(sk):
            v = (sk > bp) & (sk - bp < 6000)
            if v.sum():
                vh = sp_['peak_heights'][np.where(v)[0]]
                si = np.argmax(vh)
                bs, sc = sk[v][si], vh[si]
        if bp or bs: res.append((bp, bs, pc, sc))
    return res


def run_test(label, csv_path, h5_path, out_dir, ds_cls, model_type,
             n_samples=10000, batch_size=1024, moe_weights=None):
    print(f"\n{'='*60}")
    print(f"  {label} | {n_samples}条 | batch={batch_size}")
    print(f"{'='*60}")

    # 加载 trace names
    df = pd.read_csv(csv_path)
    names = df['trace_name'].sample(n=n_samples, random_state=36).tolist()
    print(f"  数据加载完成: {len(names)} 条")

    # 打开 HDF5 并创建 dataset
    h5 = h5py.File(h5_path, 'r')
    ds = ds_cls(h5, names)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=6,
                        pin_memory=True, prefetch_factor=2)

    # 加载模型
    dev = torch.device('cuda')
    if model_type == 'eqtransformer':
        import seisbench.models as sbm
        model = sbm.EQTransformer.from_pretrained("original").to(dev).eval()
        def unpack(o): return o[1].cpu(), o[2].cpu()
    elif model_type == 'phasenet':
        import seisbench.models as sbm
        model = sbm.PhaseNet.from_pretrained("original").to(dev).eval()
        def unpack(o): return o[:,1,:].cpu(), o[:,2,:].cpu()
    elif model_type == 'moe':
        from model_moe_gpt_PS3_CPU import IMG1
        model = IMG1()
        model.load_state_dict(torch.load(moe_weights, map_location='cpu'))
        model = model.to(dev)
        model.eval()
        def unpack(o):
            output, _ = o  # o = ((P,S), loss)
            return output[0].cpu(), output[1].cpu()

    # 预热
    for batch in loader:
        model(batch.to(dev, non_blocking=True)); break

    # 正式计时（纯推理时间）
    results = []
    torch.cuda.synchronize()
    t0 = time.time()
    infer_time = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {label.split('/')[0]}/{label.split('/')[1]}"):
            data = batch.to(dev, non_blocking=True)
            torch.cuda.synchronize()  # 等数据到GPU
            t1 = time.time()
            pp, sp = unpack(model(data))
            torch.cuda.synchronize()  # 等GPU完成
            infer_time += time.time() - t1
            bres = detect_phases_batch(pp, sp, 0.1, 0.1)
            results.extend(bres)

    total_elapsed = time.time() - t0

    # 保存
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(results, columns=['p_idx','s_idx','p_conf','s_conf']).to_csv(
        os.path.join(out_dir,'picks.csv'), index=False)
    with open(os.path.join(out_dir,'time.txt'), 'w') as f:
        f.write(str(infer_time))  # 存纯推理时间

    # 清理
    del model, loader, ds
    h5.close()
    torch.cuda.empty_cache()

    print(f"  => 检出: {len(results)}  总耗时: {total_elapsed:.1f}s  纯推理: {infer_time:.1f}s")
    return infer_time


def main():
    results = {}
    moe_w = "/home/yuan/code/SeisMoE/models/best_model_CPU.pth"

    tests = [
        # Txed (1万)
        ("Txed/EQTrans","/mnt/d/AI_Seismic_Data/txed/Texd.csv",
         "/mnt/d/AI_Seismic_Data/txed/Texd.hdf5",
         "Texd_Eqtransformer", SeismicDataset_Texd, 'eqtransformer', 10000, 1024),

        ("Txed/PhaseNet","/mnt/d/AI_Seismic_Data/txed/Texd.csv",
         "/mnt/d/AI_Seismic_Data/txed/Texd.hdf5",
         "Texd_phasenet", SeismicDataset_Texd, 'phasenet', 10000, 1024),

        ("Txed/MoE","/mnt/d/AI_Seismic_Data/txed/Texd.csv",
         "/mnt/d/AI_Seismic_Data/txed/Texd.hdf5",
         "Texd_Moe", SeismicDataset_Texd_MoE, 'moe', 10000, 1024, moe_w),

        # DiTing (1万)
        ("DiTing/EQTrans","/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.csv",
         "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.hdf5",
         "diting_Eqtransformer", SeismicDataset_DiTing, 'eqtransformer', 10000, 1024),

        ("DiTing/PhaseNet","/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.csv",
         "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.hdf5",
         "diting_PhaseNet", SeismicDataset_DiTing, 'phasenet', 10000, 1024),

        ("DiTing/MoE","/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.csv",
         "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.hdf5",
         "diting_Moe", SeismicDataset_DiTing_MoE, 'moe', 10000, 1024, moe_w),

        # Instance (10万)
        ("Instance/EQTrans","/mnt/d/AI_Seismic_Data/Instance/INSTANCE.csv",
         "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.hdf5",
         "Instance_Eqtransformer", SeismicDataset_Instance, 'eqtransformer', 100000, 1024),

        ("Instance/PhaseNet","/mnt/d/AI_Seismic_Data/Instance/INSTANCE.csv",
         "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.hdf5",
         "Instance_PhaseNet", SeismicDataset_Instance, 'phasenet', 100000, 1024),

        ("Instance/MoE","/mnt/d/AI_Seismic_Data/Instance/INSTANCE.csv",
         "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.hdf5",
         "Instance_Moe", SeismicDataset_Instance_MoE, 'moe', 100000, 1024, moe_w),

        # STEAD (10万) - 批量改小, 避免OOM
        ("STEAD/EQTrans","/mnt/d/AI_Seismic_Data/STEAD/STEAD.csv",
         "/mnt/d/AI_Seismic_Data/STEAD/STEAD.hdf5",
         "stead_Eqtransformer", SeismicDataset_STEAD, 'eqtransformer', 100000, 512),

        ("STEAD/PhaseNet","/mnt/d/AI_Seismic_Data/STEAD/STEAD.csv",
         "/mnt/d/AI_Seismic_Data/STEAD/STEAD.hdf5",
         "stead_PhaseNet", SeismicDataset_STEAD, 'phasenet', 100000, 512),

        ("STEAD/MoE","/mnt/d/AI_Seismic_Data/STEAD/STEAD.csv",
         "/mnt/d/AI_Seismic_Data/STEAD/STEAD.hdf5",
         "stead_Moe", SeismicDataset_STEAD_MoE, 'moe', 100000, 512, moe_w),
    ]

    for t in tests:
        label = t[0]
        try:
            elapsed = run_test(*t)
            results[label] = elapsed
        except Exception as e:
            print(f"  *** 失败: {e}")
            results[label] = None

    # 生成 Markdown
    md = '''# 地震波形拾取模型运行时间对比

| 数据集 | 数据量 | EQTransformer | PhaseNet | MoE |
|--------|--------|:-----------:|:--------:|:---:|
'''
    for ds in ['Txed', 'Instance', 'DiTing', 'STEAD']:
        cnt = '1万' if ds in ('Txed','DiTing') else '10万'
        cells = []
        for m in ['EQTrans','PhaseNet','MoE']:
            v = results.get(f'{ds}/{m}')
            cells.append(f'{v:.1f}s' if v else 'N/A')
        md += f"| {ds} | {cnt} | {cells[0]} | {cells[1]} | {cells[2]} |\n"

    md += '\n> GPU 8GB | 内存 64GB | CPU 24核\n'

    print(md)
    with open('/home/yuan/code/SeisMoE/benchmark/README.md','w') as f: f.write(md)
    with open('/home/yuan/code/SeisMoE/benchmark/all_times.json','w') as f: json.dump(results,f,indent=2)
    print("Markdown已保存: README.md")
    print("JSON已保存: all_times.json")


if __name__ == "__main__":
    main()
