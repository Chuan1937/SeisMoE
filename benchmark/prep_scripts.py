#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最小修改原始脚本：只改路径、样本数、MoE权重、输出目录。
不改变任何预处理、标准化、数据加载逻辑。
"""
import os, shutil

SRC = "/home/yuan/code/SeisMoE/src"
DST = "/home/yuan/code/SeisMoE/benchmark"
MOE_WEIGHTS = "/home/yuan/code/SeisMoE/models/best_model_CPU.pth"

configs = [
    # (src_file, dst_file, input_csv, hdf5_path, output_subdir, n_samples)
    # Txed - 1万
    ("seis_Texd_Eqtransformer.py", "seis_Texd_Eqtransformer.py",
     "/mnt/d/AI_Seismic_Data/txed/Texd.csv", "/mnt/d/AI_Seismic_Data/txed/Texd.hdf5",
     "Texd_Eqtransformer", 10000),
    ("seis_Texd_phasenet.py", "seis_Texd_phasenet.py",
     "/mnt/d/AI_Seismic_Data/txed/Texd.csv", "/mnt/d/AI_Seismic_Data/txed/Texd.hdf5",
     "Texd_phasenet", 10000),
    ("seis_Texd_Moe.py", "seis_Texd_Moe.py",
     "/mnt/d/AI_Seismic_Data/txed/Texd.csv", "/mnt/d/AI_Seismic_Data/txed/Texd.hdf5",
     "Texd_Moe", 10000),

    # Instance - 10万
    ("seis_Instance_Eqtransformer.py", "seis_Instance_Eqtransformer.py",
     "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.csv", "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.hdf5",
     "Instance_Eqtransformer", 100000),
    ("seis_Instance_PhaseNet.py", "seis_Instance_PhaseNet.py",
     "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.csv", "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.hdf5",
     "Instance_PhaseNet", 100000),
    ("seis_instance_Moe.py", "seis_instance_Moe.py",
     "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.csv", "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.hdf5",
     "Instance_Moe", 100000),

    # DiTing - 1万
    ("seis_diting_Eqtransformer.py", "seis_diting_Eqtransformer.py",
     "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.csv",
     "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.hdf5",
     "diting_Eqtransformer", 10000),
    ("seis_diting_PhaseNet.py", "seis_diting_PhaseNet.py",
     "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.csv",
     "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.hdf5",
     "diting_PhaseNet", 10000),
    ("seis_diting_Moe.py", "seis_diting_Moe.py",
     "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.csv",
     "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.hdf5",
     "diting_Moe", 10000),

    # STEAD - 10万
    ("seis_stead_Eqtransformer.py", "seis_stead_Eqtransformer.py",
     "/mnt/d/AI_Seismic_Data/STEAD/STEAD.csv", "/mnt/d/AI_Seismic_Data/STEAD/STEAD.hdf5",
     "stead_Eqtransformer", 100000),
    ("seis_stead_PhaseNet.py", "seis_stead_PhaseNet.py",
     "/mnt/d/AI_Seismic_Data/STEAD/STEAD.csv", "/mnt/d/AI_Seismic_Data/STEAD/STEAD.hdf5",
     "stead_PhaseNet", 100000),
    ("seis_stead_Moe.py", "seis_stead_Moe.py",
     "/mnt/d/AI_Seismic_Data/STEAD/STEAD.csv", "/mnt/d/AI_Seismic_Data/STEAD/STEAD.hdf5",
     "stead_Moe", 100000),
]

for src_name, dst_name, csv_path, h5_path, subdir, n_samples in configs:
    src = os.path.join(SRC, src_name)
    dst = os.path.join(DST, dst_name)
    
    with open(src, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. 替换 CUDA_VISIBLE_DEVICES
    import re
    content = re.sub(r'os\.environ\["CUDA_VISIBLE_DEVICES"\]\s*=\s*["\']\d+["\']',
                     'os.environ["CUDA_VISIBLE_DEVICES"]="0"', content)
    
    # 2. 替换 input_csv (不管原先是什么路径)
    content = re.sub(r'(input_csv\s*=\s*r?)(["\'])[^"\']+FileNotFoundError(["\'])',
                     f'\\1\\2{csv_path}\\3', content)
    content = re.sub(r'(input_csv\s*=\s*r?)(["\'])[^"\']+(["\'])',
                     f'\\1\\2{csv_path}\\3', content)
    
    # 3. 替换 hdf5_path
    content = re.sub(r'(hdf5_path\s*=\s*r?)(["\'])[^"\']+(["\'])',
                     f'\\1\\2{h5_path}\\3', content)
    
    # 4. 替换 output_dir
    out_dir = os.path.join(DST, subdir)
    content = re.sub(r'(output_dir\s*=\s*r?)(["\'])[^"\']+(["\'])',
                     f'\\1\\2{out_dir}\\3', content)
    
    # 5. 替换 frac 采样为固定 n_samples
    # 匹配 .sample(frac=X.XX, ...) 或 .sample(frac=X.XX)
    content = re.sub(r'\.sample\(frac=[\d.]+(,\s*random_state=\d+)?\)',
                     f'.sample(n={n_samples}, random_state=36)', content)
    
    # 6. 替换 MoE 模型权重路径
    if 'load_state_dict' in content:
        content = re.sub(r'torch\.load\(r?["\'][^"\']+best_model[^"\']*\.pth["\']',
                         f'torch.load("{MOE_WEIGHTS}"', content)
        content = re.sub(r'torch\.load\(["\'][^"\']+best_model[^"\']*\.pth["\']',
                         f'torch.load("{MOE_WEIGHTS}"', content)
    
    # 7. 替换 sys.path.append 中的 cloudlab 路径
    content = content.replace(
        'sys.path.append("/home/yuan/code/SeisMoE/src")',
        f'sys.path.append("{SRC}")'
    )
    
    # 8. (已移除 HDF5 缓存 — 保持原始 with h5py.File 模式不变)
    
    # 写入
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  Generated: {dst_name}")

print("\nDone!")
