#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成所有12个优化版本的脚本
"""

import os

# 定义所有脚本的配置
SCRIPTS_CONFIG = [
    # Txed数据集
    {
        "name": "seis_Texd_Eqtransformer",
        "input_csv": "/mnt/d/AI_Seismic_Data/txed/Texd.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/txed/Texd.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/Texd_Eqtransformer",
        "model_type": "eqtransformer",
        "model_path": None,  # 使用预训练模型
    },
    {
        "name": "seis_Texd_phasenet",
        "input_csv": "/mnt/d/AI_Seismic_Data/txed/Texd.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/txed/Texd.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/Texd_phasenet",
        "model_type": "phasenet",
        "model_path": None,
    },
    {
        "name": "seis_Texd_Moe",
        "input_csv": "/mnt/d/AI_Seismic_Data/txed/Texd.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/txed/Texd.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/Texd_Moe",
        "model_type": "moe",
        "model_path": "/home/yuan/code/SeisMoE/models/Model_Store/CPU2_moe_expert16/best_model.pth",
    },
    # Instance数据集
    {
        "name": "seis_Instance_Eqtransformer",
        "input_csv": "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/Instance_Eqtransformer",
        "model_type": "eqtransformer",
        "model_path": None,
    },
    {
        "name": "seis_Instance_PhaseNet",
        "input_csv": "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/Instance_PhaseNet",
        "model_type": "phasenet",
        "model_path": None,
    },
    {
        "name": "seis_instance_Moe",
        "input_csv": "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/Instance_Moe",
        "model_type": "moe",
        "model_path": "/home/yuan/code/SeisMoE/models/Model_Store/CPU2_moe/best_model.pth",
    },
    # Diting数据集
    {
        "name": "seis_diting_Eqtransformer",
        "input_csv": "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/diting_Eqtransformer",
        "model_type": "eqtransformer",
        "model_path": None,
    },
    {
        "name": "seis_diting_PhaseNet",
        "input_csv": "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/diting_PhaseNet",
        "model_type": "phasenet",
        "model_path": None,
    },
    {
        "name": "seis_diting_Moe",
        "input_csv": "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/diting_Moe",
        "model_type": "moe",
        "model_path": "/home/yuan/code/SeisMoE/models/Model_Store/CPU2_moe/best_model.pth",
    },
    # STEAD数据集
    {
        "name": "seis_stead_Eqtransformer",
        "input_csv": "/mnt/d/AI_Seismic_Data/STEAD/STEAD.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/STEAD/STEAD.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/stead_Eqtransformer",
        "model_type": "eqtransformer",
        "model_path": None,
    },
    {
        "name": "seis_stead_PhaseNet",
        "input_csv": "/mnt/d/AI_Seismic_Data/STEAD/STEAD.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/STEAD/STEAD.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/stead_PhaseNet",
        "model_type": "phasenet",
        "model_path": None,
    },
    {
        "name": "seis_stead_Moe",
        "input_csv": "/mnt/d/AI_Seismic_Data/STEAD/STEAD.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/STEAD/STEAD.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/stead_Moe",
        "model_type": "moe",
        "model_path": "/home/yuan/code/SeisMoE/models/Model_Store/CPU2_moe/best_model.pth",
    },
]


def generate_script(config):
    """生成优化版本的脚本"""
    
    model_import = ""
    model_load = ""
    
    if config["model_type"] == "eqtransformer":
        model_import = "import seisbench.models as sbm"
        model_load = 'model = sbm.EQTransformer.from_pretrained("original").to(device).eval()'
    elif config["model_type"] == "phasenet":
        model_import = "import seisbench.models as sbm"
        model_load = 'model = sbm.PhaseNet.from_pretrained("original").to(device).eval()'
    elif config["model_type"] == "moe":
        model_import = """import sys
sys.path.append("/home/yuan/code/SeisMoE/src")
from model_moe_gpt_PS2_CPU import IMG1"""
        model_load = f"""model = IMG1().to(device)
model.load_state_dict(torch.load("{config['model_path']}", map_location=device))
model.eval()"""
    
    # 确定输出形状处理方式
    if config["model_type"] in ["eqtransformer"]:
        output_parse = """p_prob = output[1].cpu()
            s_prob = output[2].cpu()"""
    elif config["model_type"] == "phasenet":
        output_parse = """p_prob = output[:, 1, :].cpu()
            s_prob = output[:, 2, :].cpu()"""
    elif config["model_type"] == "moe":
        output_parse = """p_prob = output[0].cpu()
            s_prob = output[1].cpu()"""
    
    script_content = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化版本 - {config["name"]}
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
{model_import}


class SeismicDataset(Dataset):
    """优化的数据集类 - 保持HDF5文件句柄打开"""
    
    def __init__(self, hdf5_path, trace_names, norm_mode='std'):
        self.hdf5_path = hdf5_path
        self.trace_names = trace_names
        self.norm_mode = norm_mode
        self.h5_file = None
        
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
            raise ValueError(f"无效的trace_name格式：{{trace_name}}") from e
        
        try:
            dataset = self.h5_file['data'].get(group)
            if dataset is None:
                raise KeyError(f"组'{{group}}'不存在")
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
        
        return {{
            'trace_name': trace_name,
            'data': torch.from_numpy(data)
        }}
    
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
    print(f"预测数据数量：{{len(trace_names)}}")
    
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
    print(f"使用设备: {{device}}")
    
    {model_load}
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="推理中"):
            data = batch['data'].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=True):
                output = model(data)
            
            {output_parse}
            
            batch_results = detect_phases_batch(p_prob, s_prob, p_thresh, s_thresh)
            
            for j, (p_idx, s_idx, p_conf, s_conf) in enumerate(batch_results):
                results.append((batch['trace_name'][j], p_idx, s_idx, p_conf, s_conf))
    
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'picks_moe2.csv')
    pd.DataFrame(results, columns=['trace_name', 'p_idx', 's_idx', 'p_conf', 's_conf']).to_csv(output_csv, index=False)
    
    print(f"拾取到的事件数：{{len(results)}}")
    print(f"结果保存到：{{output_csv}}")


if __name__ == "__main__":
    start = time.time()
    
    main(
        input_csv="{config['input_csv']}",
        hdf5_path="{config['hdf5_path']}",
        output_dir="{config['output_dir']}",
        p_thresh=0.1,
        s_thresh=0.1,
        batch_size=2048,
    )
    
    end = time.time()
    elapsed = end - start
    
    time_file = os.path.join("{config['output_dir']}", "time.txt")
    with open(time_file, 'w') as f:
        f.write(str(elapsed))
    
    print(f"运行时间: {{elapsed:.2f}}秒")
'''
    
    return script_content


def main():
    output_dir = "/home/yuan/code/SeisMoE/benchmark"
    
    for config in SCRIPTS_CONFIG:
        script_content = generate_script(config)
        script_path = os.path.join(output_dir, f"{config['name']}_optimized.py")
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print(f"生成: {script_path}")


if __name__ == "__main__":
    main()