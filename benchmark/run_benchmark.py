#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一运行器 - 顺序执行所有12个测试并记录时间
"""

import os
import random
import sys
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.signal import find_peaks
import time
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class SeismicDataset(Dataset):
    def __init__(self, hdf5_path, trace_names, norm_mode='std', mode='group'):
        self.hdf5_path = hdf5_path
        self.trace_names = trace_names
        self.norm_mode = norm_mode
        self.mode = mode
        self.h5_file = None

    def _open_hdf5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')

    def __len__(self):
        return len(self.trace_names)

    def __getitem__(self, idx):
        self._open_hdf5()
        trace_name = self.trace_names[idx]

        if self.mode == 'direct':
            data = self.h5_file['data'][trace_name][()].astype(np.float32)
        else:
            try:
                group = trace_name.split('$')[0].strip()
                index = int(trace_name.split('$')[1].split(',')[0].strip())
            except (IndexError, ValueError) as e:
                raise ValueError(f"无效的trace_name格式：'{trace_name}'") from e
            dataset = self.h5_file['data'].get(group)
            if dataset is None:
                raise KeyError(f"组'{group}'不存在")
            data = dataset[index, :, :].astype(np.float32)

        if self.norm_mode == 'std':
            mean = data.mean(axis=1, keepdims=True)
            std = data.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            data = (data - mean) / std
        elif self.norm_mode == 'max':
            data = data / (np.max(np.abs(data), axis=1, keepdims=True) + 1e-10)

        return {
            'trace_name': trace_name,
            'data': torch.from_numpy(data)
        }

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()


def detect_phases_batch(p_probs, s_probs, p_thresh=0.3, s_thresh=0.3):
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
            valid = (s_peaks > best_p) & (s_peaks - best_p < 6000)
            if valid.sum() > 0:
                idx = np.argmax(s_props['peak_heights'][np.where(valid)[0]])
                best_s = s_peaks[valid][idx]
                s_conf = s_props['peak_heights'][np.where(valid)[0]][idx]

        if best_p is not None or best_s is not None:
            results.append((best_p, best_s, p_conf, s_conf))
    return results


def run_test(config):
    """运行单个测试并返回耗时"""
    name = config['name']
    input_csv = config['input_csv']
    hdf5_path = config['hdf5_path']
    output_dir = config['output_dir']
    model_type = config['model_type']
    model_path = config.get('model_path')
    mode = config.get('mode', 'group')
    batch_size = config.get('batch_size', 1024)

    print(f"\n{'='*60}")
    print(f"运行: {name}")
    print(f"数据: {input_csv}")
    print(f"模型: {model_type}")
    print(f"{'='*60}")

    # 读取数据
    df = pd.read_csv(input_csv)
    trace_names = df['trace_name'].sample(n=10000, random_state=36).tolist()
    print(f"样本数量: {len(trace_names)}")

    dataset = SeismicDataset(hdf5_path, trace_names, mode=mode)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
    )

    device = torch.device('cuda')
    print(f"使用设备: {device}")

    # 加载模型
    if model_type == 'eqtransformer':
        import seisbench.models as sbm
        model = sbm.EQTransformer.from_pretrained("original").to(device).eval()
    elif model_type == 'phasenet':
        import seisbench.models as sbm
        model = sbm.PhaseNet.from_pretrained("original").to(device).eval()
    elif model_type == 'moe':
        from model_moe_gpt_PS2_CPU import IMG1
        model = IMG1().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

    # 计时开始
    torch.cuda.synchronize()
    t0 = time.time()

    results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="推理"):
            data = batch['data'].to(device, non_blocking=True)
            output = model(data)

            if model_type in ('eqtransformer',):
                p_prob = output[1].cpu()
                s_prob = output[2].cpu()
            elif model_type == 'phasenet':
                p_prob = output[:, 1, :].cpu()
                s_prob = output[:, 2, :].cpu()
            elif model_type == 'moe':
                p_prob = output[0].cpu()
                s_prob = output[1].cpu()

            batch_results = detect_phases_batch(p_prob, s_prob, p_thresh=0.1, s_thresh=0.1)
            for j, (p_idx, s_idx, p_conf, s_conf) in enumerate(batch_results):
                results.append((batch['trace_name'][j], p_idx, s_idx, p_conf, s_conf))

    torch.cuda.synchronize()
    elapsed = time.time() - t0

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'picks_moe2.csv')
    pd.DataFrame(results, columns=['trace_name', 'p_idx', 's_idx', 'p_conf', 's_conf']).to_csv(output_csv, index=False)

    time_file = os.path.join(output_dir, 'time.txt')
    with open(time_file, 'w') as f:
        f.write(str(elapsed))

    print(f"拾取事件: {len(results)}")
    print(f"运行时间: {elapsed:.2f} 秒")

    # 清理
    del model, loader, dataset
    torch.cuda.empty_cache()

    return elapsed


def main():
    configs = [
        # Txed数据集
        dict(name="Txed_EQTransformer",  input_csv="/mnt/d/AI_Seismic_Data/txed/Texd.csv",
             hdf5_path="/mnt/d/AI_Seismic_Data/txed/Texd.hdf5",
             output_dir="/home/yuan/code/SeisMoE/benchmark/Texd_Eqtransformer",
             model_type="eqtransformer"),
        dict(name="Txed_PhaseNet",       input_csv="/mnt/d/AI_Seismic_Data/txed/Texd.csv",
             hdf5_path="/mnt/d/AI_Seismic_Data/txed/Texd.hdf5",
             output_dir="/home/yuan/code/SeisMoE/benchmark/Texd_phasenet",
             model_type="phasenet"),
        dict(name="Txed_MoE",            input_csv="/mnt/d/AI_Seismic_Data/txed/Texd.csv",
             hdf5_path="/mnt/d/AI_Seismic_Data/txed/Texd.hdf5",
             output_dir="/home/yuan/code/SeisMoE/benchmark/Texd_Moe",
             model_type="moe",
             model_path="/home/yuan/code/SeisMoE/models/Model_Store/CPU2_moe_expert16/best_model.pth"),

        # Instance数据集
        dict(name="Instance_EQTransformer", input_csv="/mnt/d/AI_Seismic_Data/Instance/INSTANCE.csv",
             hdf5_path="/mnt/d/AI_Seismic_Data/Instance/INSTANCE.hdf5",
             output_dir="/home/yuan/code/SeisMoE/benchmark/Instance_Eqtransformer",
             model_type="eqtransformer"),
        dict(name="Instance_PhaseNet",      input_csv="/mnt/d/AI_Seismic_Data/Instance/INSTANCE.csv",
             hdf5_path="/mnt/d/AI_Seismic_Data/Instance/INSTANCE.hdf5",
             output_dir="/home/yuan/code/SeisMoE/benchmark/Instance_PhaseNet",
             model_type="phasenet"),
        dict(name="Instance_MoE",           input_csv="/mnt/d/AI_Seismic_Data/Instance/INSTANCE.csv",
             hdf5_path="/mnt/d/AI_Seismic_Data/Instance/INSTANCE.hdf5",
             output_dir="/home/yuan/code/SeisMoE/benchmark/Instance_Moe",
             model_type="moe",
             model_path="/home/yuan/code/SeisMoE/models/Model_Store/CPU2_moe/best_model.pth"),

        # DiTing数据集
        dict(name="DiTing_EQTransformer", input_csv="/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.csv",
             hdf5_path="/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.hdf5",
             output_dir="/home/yuan/code/SeisMoE/benchmark/diting_Eqtransformer",
             model_type="eqtransformer", mode="direct"),
        dict(name="DiTing_PhaseNet",      input_csv="/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.csv",
             hdf5_path="/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.hdf5",
             output_dir="/home/yuan/code/SeisMoE/benchmark/diting_PhaseNet",
             model_type="phasenet", mode="direct"),
        dict(name="DiTing_MoE",           input_csv="/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.csv",
             hdf5_path="/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.hdf5",
             output_dir="/home/yuan/code/SeisMoE/benchmark/diting_Moe",
             model_type="moe", mode="direct",
             model_path="/home/yuan/code/SeisMoE/models/Model_Store/CPU2_moe/best_model.pth"),

        # STEAD数据集
        dict(name="STEAD_EQTransformer", input_csv="/mnt/d/AI_Seismic_Data/STEAD/STEAD.csv",
             hdf5_path="/mnt/d/AI_Seismic_Data/STEAD/STEAD.hdf5",
             output_dir="/home/yuan/code/SeisMoE/benchmark/stead_Eqtransformer",
             model_type="eqtransformer"),
        dict(name="STEAD_PhaseNet",      input_csv="/mnt/d/AI_Seismic_Data/STEAD/STEAD.csv",
             hdf5_path="/mnt/d/AI_Seismic_Data/STEAD/STEAD.hdf5",
             output_dir="/home/yuan/code/SeisMoE/benchmark/stead_PhaseNet",
             model_type="phasenet"),
        dict(name="STEAD_MoE",           input_csv="/mnt/d/AI_Seismic_Data/STEAD/STEAD.csv",
             hdf5_path="/mnt/d/AI_Seismic_Data/STEAD/STEAD.hdf5",
             output_dir="/home/yuan/code/SeisMoE/benchmark/stead_Moe",
             model_type="moe",
             model_path="/home/yuan/code/SeisMoE/models/Model_Store/CPU2_moe/best_model.pth"),
    ]

    results = {}
    for cfg in configs:
        try:
            elapsed = run_test(cfg)
            results[cfg['name']] = elapsed
        except Exception as e:
            print(f"  *** 失败: {e}")
            results[cfg['name']] = None

    # 保存JSON
    with open('/home/yuan/code/SeisMoE/benchmark/all_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 生成Markdown
    md = generate_markdown(results)
    with open('/home/yuan/code/SeisMoE/benchmark/README.md', 'w') as f:
        f.write(md)

    print("\n" + md)


def generate_markdown(results):
    datasets = ['Txed', 'Instance', 'DiTing', 'STEAD']
    models = ['EQTransformer', 'PhaseNet', 'MoE']

    md = "# 地震波形拾取模型运行时间对比\n\n"
    md += "## 测试环境\n"
    md += "- **CPU**: 24核 | **内存**: 64GB | **GPU**: 8GB显存\n"
    md += "- **数据量**: 每个数据集10,000条波形\n\n"
    md += "## 运行时间（秒）\n\n"
    md += "| 数据集 | EQTransformer | PhaseNet | MoE |\n"
    md += "|--------|:-----------:|:--------:|:---:|\n"

    for ds in datasets:
        row = f"| {ds} |"
        for m in models:
            key = f"{ds}_{m}"
            val = results.get(key)
            if val is not None:
                row += f" {val:.2f} |"
            else:
                row += " N/A |"
        md += row + "\n"

    md += "\n## 说明\n"
    md += "- 每个测试使用10,000条随机采样的波形\n"
    md += "- 使用 `torch.cuda.synchronize()` 确保GPU操作完成后再计时\n"
    md += "- 每个测试完成后清除GPU缓存\n"
    md += "- 批处理大小: 1024\n"
    return md


if __name__ == "__main__":
    main()
