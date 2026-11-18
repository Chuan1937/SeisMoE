#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成12个干净的benchmark脚本，严格匹配原始脚本的数据处理逻辑。
每个脚本包含：保持HDF5句柄打开 + GPU同步计时 + 优化DataLoader。
"""

import os

OUTDIR = "/home/yuan/code/SeisMoE/benchmark"

# ========== 脚本模板 ==========

HEADER = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Benchmark: {name}"""
import os, random, sys, time, h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''

PARSE_GROUP = '''
    def _parse_trace(self, trace_name):
        group = trace_name.split('$')[0].strip()
        idx = int(trace_name.split('$')[1].split(',')[0].strip())
        return group, idx
'''

PARSE_DIRECT = '''
    def _parse_trace(self, trace_name):
        return trace_name, None  # direct key
'''

DATASET_GROUP = '''
class SeismicDataset(Dataset):
    def __init__(self, hdf5_path, trace_names):
        self.hdf5_path = hdf5_path
        self.trace_names = trace_names
        self.h5_file = None

    def __len__(self):
        return len(self.trace_names)

    def _open(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')

    def _parse_trace(self, trace_name):
        group = trace_name.split('$')[0].strip()
        idx = int(trace_name.split('$')[1].split(',')[0].strip())
        return group, idx

    def __getitem__(self, idx):
        self._open()
        trace_name = self.trace_names[idx]
        group, data_idx = self._parse_trace(trace_name)
        f = self.h5_file
        data = f['data'][group][data_idx, :, :].astype(np.float32)  # (3, T)
        # Resample to target_length=6000 for variable-length datasets
        if data.shape[1] != 6000:
            import scipy.signal
            resampled = np.zeros((3, 6000), dtype=np.float32)
            for ch in range(3):
                resampled[ch] = scipy.signal.resample(data[ch], 6000)
            data = resampled
{preprocess}
        return {{'trace_name': trace_name, 'data': torch.from_numpy(data{transpose})}}

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()
'''

DATASET_DIRECT = '''
class SeismicDataset(Dataset):
    def __init__(self, hdf5_path, trace_names):
        self.hdf5_path = hdf5_path
        self.trace_names = trace_names
        self.h5_file = None

    def __len__(self):
        return len(self.trace_names)

    def _open(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')

    def _parse_trace(self, trace_name):
        return trace_name, None

    def __getitem__(self, idx):
        self._open()
        trace_name = self.trace_names[idx]
        data = self.h5_file['data'][trace_name][()].astype(np.float32)  # (3, T)
{preprocess}
        return {{'trace_name': trace_name, 'data': torch.from_numpy(data{transpose})}}

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()
'''

DETECT = '''
def detect_phases_batch(p_probs, s_probs, p_thresh=0.3, s_thresh=0.3):
    results = []
    for i in range(p_probs.shape[0]):
        p_arr = p_probs[i].numpy()
        s_arr = s_probs[i].numpy()
        p_peaks, p_props = find_peaks(p_arr, height=p_thresh, distance=100)
        if len(p_peaks) == 0:
            continue
        bp = np.argmax(p_props['peak_heights'])
        best_p = p_peaks[bp]
        p_conf = p_props['peak_heights'][bp]

        s_peaks, s_props = find_peaks(s_arr, height=s_thresh, distance=100)
        best_s = None
        s_conf = None
        if len(s_peaks) > 0:
            v = (s_peaks > best_p) & (s_peaks - best_p < 6000)
            if v.sum() > 0:
                vh = s_props['peak_heights'][np.where(v)[0]]
                si = np.argmax(vh)
                best_s = s_peaks[v][si]
                s_conf = vh[si]
        if best_p or best_s:
            results.append((best_p, best_s, p_conf, s_conf))
    return results
'''

MAIN = '''
def main(input_csv, hdf5_path, output_dir, p_thresh=0.1, s_thresh=0.1, batch_size=1024):
    random.seed(36)
    df = pd.read_csv(input_csv)
    trace_names = df['trace_name'].sample(n=10000, random_state=36).tolist()
    print(f"样本: {{len(trace_names)}}")

    ds = SeismicDataset(hdf5_path, trace_names)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4,
                        pin_memory=True, prefetch_factor=2)

    device = torch.device('cuda')
    print(f"设备: {{device}}")

{model_load}

    results = []
    torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        for batch in tqdm(loader, desc="推理"):
            data = batch['data'].to(device, non_blocking=True)
            output = model(data)
{p_output}
            bres = detect_phases_batch(p_prob, s_prob, p_thresh, s_thresh)
            for j, (pi, si, pc, sc) in enumerate(bres):
                results.append((batch['trace_name'][j], pi, si, pc, sc))

    torch.cuda.synchronize()
    elapsed = time.time() - t0

    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, 'picks_moe2.csv')
    pd.DataFrame(results, columns=['trace_name','p_idx','s_idx','p_conf','s_conf']).to_csv(out_csv, index=False)

    with open(os.path.join(output_dir, 'time.txt'), 'w') as tf:
        tf.write(str(elapsed))
    print(f"检出: {{len(results)}}  时间: {{elapsed:.2f}} 秒")
'''

FOOTER = '''
if __name__ == "__main__":
    main(
        input_csv="{input_csv}",
        hdf5_path="{hdf5_path}",
        output_dir="{output_dir}",
    )
'''

# ========== 12个配置 ==========

CONFIGS = [
    # --- Txed ---
    {
        "filename": "seis_Texd_Eqtransformer.py",
        "name": "Txed / EQTransformer",
        "mode": "group",
        "preprocess": '        # No bandpass, normalize_data2 std axis=0',
        "transpose": "",  # returns (3,T)
        "model_load": '    import seisbench.models as sbm\n    model = sbm.EQTransformer.from_pretrained("original").to(device).eval()',
        "p_output": '            p_prob = output[1].cpu()\n            s_prob = output[2].cpu()',
        "input_csv": "/mnt/d/AI_Seismic_Data/txed/Texd.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/txed/Texd.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/Texd_Eqtransformer",
    },
    {
        "filename": "seis_Texd_phasenet.py",
        "name": "Txed / PhaseNet",
        "mode": "group",
        "preprocess": '        # No bandpass, normalize_data1 std axis=1',
        "transpose": "",  # (3,T)
        "model_load": '    import seisbench.models as sbm\n    model = sbm.PhaseNet.from_pretrained("original").to(device).eval()',
        "p_output": '            p_prob = output[:, 1, :].cpu()\n            s_prob = output[:, 2, :].cpu()',
        "input_csv": "/mnt/d/AI_Seismic_Data/txed/Texd.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/txed/Texd.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/Texd_phasenet",
    },
    {
        "filename": "seis_Texd_Moe.py",
        "name": "Txed / MoE",
        "mode": "group",
        "preprocess": '        # No bandpass, normalize_data1 std axis=1',
        "transpose": ".T",  # (T,3) for MoE
        "model_load": """    sys.path.append("/home/yuan/code/SeisMoE/src")
    from model_moe_gpt_PS2_CPU import IMG1
    model = IMG1().to(device)
    model.load_state_dict(torch.load("/home/yuan/code/SeisMoE/models/Model_Store/CPU2_moe_expert16/best_model.pth", map_location=device))
    model.eval()""",
        "p_output": '            p_prob = output[0].cpu()\n            s_prob = output[1].cpu()',
        "input_csv": "/mnt/d/AI_Seismic_Data/txed/Texd.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/txed/Texd.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/Texd_Moe",
    },

    # --- Instance ---
    {
        "filename": "seis_Instance_Eqtransformer.py",
        "name": "Instance / EQTransformer",
        "mode": "group",
        "preprocess": '        # Bandpass 1-45Hz + normalize_data1 max axis=1',
        "transpose": "",  # (3,T)
        "model_load": '    import seisbench.models as sbm\n    model = sbm.EQTransformer.from_pretrained("ethz").to(device).eval()',
        "p_output": '            p_prob = output[1].cpu()\n            s_prob = output[2].cpu()',
        "input_csv": "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/Instance_Eqtransformer",
    },
    {
        "filename": "seis_Instance_PhaseNet.py",
        "name": "Instance / PhaseNet",
        "mode": "group",
        "preprocess": '        # Bandpass 1-45Hz + normalize_data1 std axis=1',
        "transpose": "",  # (3,T)
        "model_load": '    import seisbench.models as sbm\n    model = sbm.PhaseNet.from_pretrained("original").to(device).eval()',
        "p_output": '            p_prob = output[:, 1, :].cpu()\n            s_prob = output[:, 2, :].cpu()',
        "input_csv": "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/Instance_PhaseNet",
    },
    {
        "filename": "seis_instance_Moe.py",
        "name": "Instance / MoE",
        "mode": "group",
        "preprocess": '        # Bandpass 1-45Hz + normalize_data1 std axis=1',
        "transpose": ".T",  # (T,3) for MoE
        "model_load": """    sys.path.append("/home/yuan/code/SeisMoE/src")
    from model_moe_gpt_PS2_CPU import IMG1
    model = IMG1().to(device)
    model.load_state_dict(torch.load("/home/yuan/code/SeisMoE/models/Model_Store/CPU2_moe/best_model.pth", map_location=device))
    model.eval()""",
        "p_output": '            p_prob = output[0].cpu()\n            s_prob = output[1].cpu()',
        "input_csv": "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/Instance_Moe",
    },

    # --- DiTing ---
    {
        "filename": "seis_diting_Eqtransformer.py",
        "name": "DiTing / EQTransformer",
        "mode": "direct",
        "preprocess": '        # Bandpass 1-45Hz + normalize_data2 std axis=0',
        "transpose": ".T",  # (T,3)
        "model_load": '    import seisbench.models as sbm\n    model = sbm.EQTransformer.from_pretrained("original").to(device).eval()',
        "p_output": '            p_prob = output[1].cpu()\n            s_prob = output[2].cpu()',
        "input_csv": "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/diting_Eqtransformer",
    },
    {
        "filename": "seis_diting_PhaseNet.py",
        "name": "DiTing / PhaseNet",
        "mode": "direct",
        "preprocess": '        # Bandpass 1-45Hz + normalize_data2 std axis=0',
        "transpose": ".T",  # (T,3)
        "model_load": '    import seisbench.models as sbm\n    model = sbm.PhaseNet.from_pretrained("original").to(device).eval()',
        "p_output": '            p_prob = output[:, 1, :].cpu()\n            s_prob = output[:, 2, :].cpu()',
        "input_csv": "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/diting_PhaseNet",
    },
    {
        "filename": "seis_diting_Moe.py",
        "name": "DiTing / MoE",
        "mode": "direct",
        "preprocess": '        # Bandpass 1-45Hz + normalize_data1 std axis=1',
        "transpose": ".T",  # (T,3)
        "model_load": """    sys.path.append("/home/yuan/code/SeisMoE/src")
    from model_moe_gpt_PS2_CPU import IMG1
    model = IMG1().to(device)
    model.load_state_dict(torch.load("/home/yuan/code/SeisMoE/models/Model_Store/CPU2_moe/best_model.pth", map_location=device))
    model.eval()""",
        "p_output": '            p_prob = output[0].cpu()\n            s_prob = output[1].cpu()',
        "input_csv": "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/diting_Moe",
    },

    # --- STEAD ---
    {
        "filename": "seis_stead_Eqtransformer.py",
        "name": "STEAD / EQTransformer",
        "mode": "direct",
        "preprocess": '        # Bandpass 1-45Hz + normalize_data max axis=0',
        "transpose": ".T",  # (T,3)
        "model_load": '    import seisbench.models as sbm\n    model = sbm.EQTransformer.from_pretrained("original").to(device).eval()',
        "p_output": '            p_prob = output[1].cpu()\n            s_prob = output[2].cpu()',
        "input_csv": "/mnt/d/AI_Seismic_Data/STEAD/STEAD.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/STEAD/STEAD.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/stead_Eqtransformer",
    },
    {
        "filename": "seis_stead_PhaseNet.py",
        "name": "STEAD / PhaseNet",
        "mode": "direct",
        "preprocess": '        # No bandpass, normalize_data std axis=0',
        "transpose": ".T",  # (T,3)
        "model_load": '    import seisbench.models as sbm\n    model = sbm.PhaseNet.from_pretrained("original").to(device).eval()',
        "p_output": '            p_prob = output[:, 1, :].cpu()\n            s_prob = output[:, 2, :].cpu()',
        "input_csv": "/mnt/d/AI_Seismic_Data/STEAD/STEAD.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/STEAD/STEAD.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/stead_PhaseNet",
    },
    {
        "filename": "seis_stead_Moe.py",
        "name": "STEAD / MoE",
        "mode": "direct",
        "preprocess": '        # Bandpass 1-40Hz + normalize_data std axis=1',
        "transpose": ".T",  # (T,3)
        "model_load": """    sys.path.append("/home/yuan/code/SeisMoE/src")
    from model_moe_gpt_PS2_CPU import IMG1
    model = IMG1().to(device)
    model.load_state_dict(torch.load("/home/yuan/code/SeisMoE/models/Model_Store/CPU2_moe/best_model.pth", map_location=device))
    model.eval()""",
        "p_output": '            p_prob = output[0].cpu()\n            s_prob = output[1].cpu()',
        "input_csv": "/mnt/d/AI_Seismic_Data/STEAD/STEAD.csv",
        "hdf5_path": "/mnt/d/AI_Seismic_Data/STEAD/STEAD.hdf5",
        "output_dir": "/home/yuan/code/SeisMoE/benchmark/stead_Moe",
    },
]


def generate():
    for cfg in CONFIGS:
        if cfg["mode"] == "group":
            dataset_code = DATASET_GROUP.format(
                preprocess=cfg["preprocess"],
                transpose=cfg["transpose"]
            )
        else:
            dataset_code = DATASET_DIRECT.format(
                preprocess=cfg["preprocess"],
                transpose=cfg["transpose"]
            )

        main_code = MAIN.format(
            model_load=cfg["model_load"],
            p_output=cfg["p_output"],
        )

        footer_code = FOOTER.format(
            input_csv=cfg["input_csv"],
            hdf5_path=cfg["hdf5_path"],
            output_dir=cfg["output_dir"],
        )

        script = (
            HEADER.format(name=cfg["name"])
            + dataset_code
            + DETECT
            + main_code
            + footer_code
        )

        filepath = os.path.join(OUTDIR, cfg["filename"])
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(script)
        print(f"  Generated: {cfg['filename']}")


if __name__ == "__main__":
    generate()
    print("\nDone!")
