import os
import time
import csv
import h5py
import torch
import numpy as np
import pandas as pd
import random
import gc
import sys
import multiprocessing as mp
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
import seisbench.models as sbm
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

###############################################################
# 1) 与原代码相同的函数 (picker, _detect_peaks, _get_snr, prediction_writer)
###############################################################
from predict_subfunction_DPS import _detect_peaks, prediction_writer, picker, _get_snr

###############################################################
# 2) 改写 Dataset，不在 __init__ 持有 h5py.File
###############################################################
class SeismicDataset(Dataset):
    def __init__(self, hdf5_file, trace_names, normalization_mode='std'):
        self.hdf5_path = hdf5_file
        self.trace_names = trace_names
        self.normalization_mode = normalization_mode

    def __len__(self):
        return len(self.trace_names)

    def __getitem__(self, idx):
        trace_name = self.trace_names[idx]
        with h5py.File(self.hdf5_path, 'r') as f:
            data = np.array(f['data'][trace_name], dtype=np.float32)

        if self.normalization_mode == 'std':
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-10)
        elif self.normalization_mode == 'max':
            data = data / (np.max(np.abs(data), axis=0) + 1e-10)

        # EQTransformer 模型需要 (3, N)
        data = data.T
        data_torch = torch.from_numpy(data)
        return {
            'input': data_torch,
            'trace_name': trace_name
        }

###############################################################
# 3) 多进程并行后处理（含进度条）
###############################################################
def process_sublist(
    sublist,
    hdf5_file,
    prob_dic,
    detection_threshold,
    P_threshold,
    S_threshold,
    partial_csv_prefix
):
    """
    每个子进程的工作：
      - 读取 sublist: [(trace_name, idx_array), ...]
      - 对每条记录执行: (1) 读取波形  (2) picker(...) (3) SNR + 2秒内去重 + 写 partial CSV
      - 通过 tqdm 显示子进程进度, desc 显示 PID.
    """
    import h5py
    partial_csv = partial_csv_prefix + f".pid_{os.getpid()}.csv"

    # 写 CSV 表头
    with open(partial_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'file_name', 'network', 'station', 'instrument_type',
            'station_lat', 'station_lon', 'station_elv',
            'event_start_time', 'event_end_time',
            'detection_probability', 'detection_uncertainty',
            'p_arrival_time', 'p_probability', 'p_uncertainty', 'p_snr',
            's_arrival_time', 's_probability', 's_uncertainty', 's_snr'
        ])

    detection_memory_local = []

    with h5py.File(hdf5_file, 'r') as h5, open(partial_csv, 'a', newline='') as f:
        writer = csv.writer(f)

        # 加进度条
        for (trace_name, idx_ar) in tqdm(
            sublist,
            desc=f"Worker {os.getpid()}",
            total=len(sublist),
            leave=True
        ):
            dat = np.array(h5['data'][trace_name])
            # 归一化
            dat = (dat - np.mean(dat, axis=0)) / (np.std(dat, axis=0) + 1e-10)

            # 取该记录预测好的 (D, P, S) 概率
            dd_mean = prob_dic['DD_mean'][idx_ar]
            pp_mean = prob_dic['PP_mean'][idx_ar]
            ss_mean = prob_dic['SS_mean'][idx_ar]
            dd_std  = prob_dic['DD_std'][idx_ar]
            pp_std  = prob_dic['PP_std'][idx_ar]
            ss_std  = prob_dic['SS_std'][idx_ar]

            # 调用 picker
            matches, _, _ = picker(
                detection_threshold, P_threshold, S_threshold,
                dd_mean, pp_mean, ss_mean,
                dd_std, pp_std, ss_std
            )
            # 只处理第一个 match
            if len(matches) >= 1:
                first_key = list(matches)[0]
                if matches[first_key][3] and matches[first_key][6]:
                    if (matches[first_key][6] - matches[first_key][3]) < 60 * 100:
                        # 计算 SNR
                        snr_val = [
                            _get_snr(dat, matches[first_key][3], window=100),
                            _get_snr(dat, matches[first_key][6], window=100)
                        ]
                        dataset_trace = h5['data'][trace_name]
                        detection_memory_local = prediction_writer(
                            dataset_trace, writer, f, matches, snr_val,
                            detection_memory_local
                        )

    return partial_csv

def process_predictions_parallel(
    batch_results,
    hdf5_file,
    prob_dic,
    detection_threshold,
    P_threshold,
    S_threshold,
    partial_csv_prefix
):
    """
    并行调度：将 batch_results 分块给多个进程，写 partial CSV
    """
    import math
    from functools import partial

    n_proc = min(8, mp.cpu_count())  # 并行进程数
    chunk_size = math.ceil(len(batch_results) / n_proc)
    chunks = [batch_results[i:i+chunk_size] for i in range(0, len(batch_results), chunk_size)]

    with mp.Pool(n_proc) as pool:
        func = partial(
            process_sublist,
            hdf5_file=hdf5_file,
            prob_dic=prob_dic,
            detection_threshold=detection_threshold,
            P_threshold=P_threshold,
            S_threshold=S_threshold,
            partial_csv_prefix=partial_csv_prefix
        )
        partial_csv_list = pool.map(func, chunks)

    return partial_csv_list

def final_deduplicate(partial_csv_list, final_csv_path):
    """
    合并所有 partial CSV，然后做一次全局 2 秒去重
    """
    dfs = []
    for pcsv in partial_csv_list:
        if os.path.exists(pcsv):
            dfs.append(pd.read_csv(pcsv))
    if len(dfs) == 0:
        pd.DataFrame().to_csv(final_csv_path, index=False)
        return 0

    big_df = pd.concat(dfs, ignore_index=True)
    # 按 event_start_time 排序
    big_df['event_start_time'] = pd.to_datetime(big_df['event_start_time'], errors='coerce')
    big_df.sort_values(by='event_start_time', inplace=True)

    deduped_rows = []
    last_time = None
    for _, row in big_df.iterrows():
        cur_time = row['event_start_time']
        if pd.isnull(cur_time):
            deduped_rows.append(row)
            continue
        if last_time is None:
            deduped_rows.append(row)
            last_time = cur_time
        else:
            delta_sec = (cur_time - last_time).total_seconds()
            if delta_sec >= 2:
                deduped_rows.append(row)
                last_time = cur_time
            else:
                pass

    final_df = pd.DataFrame(deduped_rows)
    final_df.to_csv(final_csv_path, index=False)
    return len(final_df)

###############################################################
# 4) 主函数 eqt_prediction 并行后处理 (带CPU多进程筛选+进度条)
###############################################################
def eqt_prediction_parallel(
    input_csv,
    input_hdf5,
    store_dir,
    detection_threshold=0.5,
    P_threshold=0.3,
    S_threshold=0.3,
    batch_size=20,
    device=None,
    random_seed=42,
    ratio=0.1
):
    """
    1) 用 EQTransformer 做推理 => 得到 prob_dic
    2) 构造 batch_results => 多进程并行 CPU 后处理 => partial CSV + 进度条
    3) 最后合并 + 2秒去重 => final CSV
    """
    start_time = time.time()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}')
    # ====== (A) 加载 EQTransformer ======
    model = sbm.EQTransformer.from_pretrained("stead").to(device)
    model.eval()

    # ====== (B) 随机抽样
    df_csv = pd.read_csv(input_csv, low_memory=False)
    full_list = df_csv["trace_name"].tolist()
    random.seed(random_seed)
    total_samples = len(full_list)
    sample_size = int(total_samples * ratio)
    predict_list = random.sample(full_list, sample_size)
    print(f"Total {total_samples} traces, sample => {len(predict_list)} for eqt_prediction")

    # ====== (C) GPU/CPU 推理 => 得到 (D, P, S)
    dataset = SeismicDataset(input_hdf5, predict_list, normalization_mode='std')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    all_det, all_p, all_s = [], [], []
    print("Start predicting (EQTransformer) ...")
    with torch.no_grad():
        for batch in tqdm(loader):
            inputs = batch['input'].float().to(device)
            outputs = model(inputs)  # => [det, p, s], each shape (B, N)
            all_det.append(outputs[0].cpu().numpy())
            all_p.append(outputs[1].cpu().numpy())
            all_s.append(outputs[2].cpu().numpy())

    all_det = np.concatenate(all_det, axis=0)
    all_p   = np.concatenate(all_p,   axis=0)
    all_s   = np.concatenate(all_s,   axis=0)

    prob_dic = {
        'DD_mean': all_det,
        'PP_mean': all_p,
        'SS_mean': all_s,
        'DD_std': np.zeros_like(all_det),
        'PP_std': np.zeros_like(all_p),
        'SS_std': np.zeros_like(all_s),
    }

    del all_det, all_p, all_s
    gc.collect()

    # ====== (D) 并行后处理
    # 构造 batch_results = [(trace_name, idx), ...]
    batch_results = [(tn, i) for i, tn in enumerate(predict_list)]

    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    partial_csv_prefix = os.path.join(store_dir, f"STEAD_EQTransformer_Predict_{random_seed}")

    print("Start parallel post-processing (with tqdm for each worker) ...")
    partial_csv_list = process_predictions_parallel(
        batch_results=batch_results,
        hdf5_file=input_hdf5,
        prob_dic=prob_dic,
        detection_threshold=detection_threshold,
        P_threshold=P_threshold,
        S_threshold=S_threshold,
        partial_csv_prefix=partial_csv_prefix
    )

    # ====== (E) 最终合并 + 2秒去重
    final_csv_path = partial_csv_prefix + "_final.csv"
    final_count = final_deduplicate(partial_csv_list, final_csv_path)

    # ====== (F) 打印统计
    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"\n*** Finished eqt_prediction & parallel post-processing in {int(hours)}h {int(minutes)}m {round(seconds, 2)}s.")
    print(f"*** Final deduplicated events: {final_count}")
    print(f"*** CSV => {final_csv_path}")
    sys.exit(0)

#########################################################
# 测试入口
#########################################################
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    input_csv = "/mnt/nn/Code/DATA/STEAD/STEAD.csv"
    input_hdf5 = "/mnt/nn/Code/DATA/STEAD/STEAD.hdf5"
    store_dir = "/home/yuan/code/SeisMoE/data/test_parallel"

    eqt_prediction_parallel(
        input_csv=input_csv,
        input_hdf5=input_hdf5,
        store_dir=store_dir,
        detection_threshold=0.5,
        P_threshold=0.3,
        S_threshold=0.3,
        batch_size=200,
        device=None,     # CPU或GPU均可
        random_seed=81,
        ratio=0.1
    )
