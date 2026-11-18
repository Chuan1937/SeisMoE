import os
import csv
import math
import time
import h5py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
import seisbench.models as sbm
import multiprocessing as mp
import random
import gc

############################################
# predict_subfunction_PS.py 中的函数直接复制
############################################
def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):
    """
    Identical to your existing _detect_peaks. 
    """
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    dx = x[1:] - x[:-1]

    # Handle NaNs
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf

    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]

    ind = np.unique(np.hstack((ine, ire, ife)))

    # Filter out neighbors of NaNs
    if ind.size and indnan.size:
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # Remove edge peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # Minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # Threshold
    if ind.size and threshold > 0:
        dx_min = np.min(np.vstack([x[ind] - x[ind-1], x[ind] - x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx_min < threshold)[0])
    # Enforce mpd
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                idel = idel | ((ind >= ind[i]-mpd) & (ind <= ind[i]+mpd) 
                               & (x[ind[i]] > x[ind] if kpsh else True))
                idel[i] = 0
        ind = np.sort(ind[~idel])
    return ind

# def phasenet_picker(P_threshold, S_threshold, p_array, s_array):
#     """
#     修改后:
#       只保留 p_array 中概率最高的那个 P 拾取，
#       在其后 60s (可自行改) 内寻找第一个符合条件的 S 拾取。
#       若没找到 S，则只输出 P。
#       这样对于一个 trace 只会输出一个事件。
    
#     返回形式:
#         { best_p_idx: (best_s_idx or None, best_p_prob, best_s_prob or None) }
#     """
#     # 1) 找所有 P 的峰值
#     p_inds = _detect_peaks(p_array, mph=P_threshold, mpd=1)
#     if len(p_inds) == 0:
#         # 没有任何 P，返回空
#         return {}
#     # 2) 找到概率最高的 P
#     best_p_idx = None
#     best_p_prob = -999
#     for p_idx in p_inds:
#         prob = p_array[p_idx]
#         if prob > best_p_prob:
#             best_p_idx = p_idx
#             best_p_prob = prob

#     # 3) 在 60s 内找第一个符合门槛的 S
#     s_inds = _detect_peaks(s_array, mph=S_threshold, mpd=1)
#     best_s_idx = None
#     best_s_prob = None

#     p_time_sec = best_p_idx / 100.0
#     for s_idx in s_inds:
#         s_time_sec = s_idx / 100.0
#         if 0 < (s_time_sec - p_time_sec) < 60:
#             best_s_idx = s_idx
#             best_s_prob = s_array[s_idx]
#             break  # 只取第一个符合的 S

#     # 4) 组装返回
#     matches = {}
#     if best_s_idx is not None:
#         matches[best_p_idx] = (best_s_idx, best_p_prob, best_s_prob)
#     else:
#         # 如果想只保留「P+S 组合」则不写
#         # 但此处我们保留仅 P 的结果
#         matches[best_p_idx] = (None, best_p_prob, None)

#     return matches

def phasenet_picker(P_threshold, S_threshold, p_array, s_array):
    matches = {}
    
    # 1) 处理 P 波拾取
    p_inds = _detect_peaks(p_array, mph=P_threshold, mpd=1)
    if len(p_inds) > 0:  # 正确判断：检查数组是否非空
        # 存在 P 波：找到概率最高的 P
        best_p_idx = p_inds[p_array[p_inds].argmax()]  # 更高效的写法
        best_p_prob = p_array[best_p_idx]
        
        # 在 P 波后 60s 内寻找 S 波
        s_inds = _detect_peaks(s_array, mph=S_threshold, mpd=1)
        best_s_idx = None
        best_s_prob = None
        p_time_sec = best_p_idx / 100.0
        for s_idx in s_inds:
            s_time_sec = s_idx / 100.0
            if 0 < (s_time_sec - p_time_sec) < 60:
                best_s_idx = s_idx
                best_s_prob = s_array[s_idx]
                break
        
        # 保存 P 波（无论是否有 S）
        matches[best_p_idx] = (best_s_idx, best_p_prob, best_s_prob)
    else:
        # 2) 没有 P 波时：处理 S 波拾取
        s_inds = _detect_peaks(s_array, mph=S_threshold, mpd=1)
        if len(s_inds) > 0:  # 正确判断
            # 找到概率最高的 S
            best_s_idx = s_inds[s_array[s_inds].argmax()]
            best_s_prob = s_array[best_s_idx]
            # 保存 S 波（无 P）
            matches[None] = (best_s_idx, None, best_s_prob)
    
    return matches
def _get_snr(data, pick_index, window=200):
    """
    计算 SNR
    """
    snr = None
    if pick_index is not None:
        try:
            pick_index = int(pick_index)
            nstart = max(0, pick_index - window)
            nend   = pick_index
            sstart = pick_index
            send   = min(pick_index + window, len(data))
            noise  = data[nstart:nend]
            signal = data[sstart:send]
            if len(noise) > 0 and len(signal) > 0:
                snr_val = (np.percentile(signal, 95) / np.percentile(noise, 95))**2
                snr = round(10 * math.log10(snr_val), 1)
        except:
            pass
    return snr

def phasenet_prediction_writer(
    dataset_h5,
    trace_name,
    p_idx,
    s_idx,
    p_prob,
    s_prob,
    csv_writer,
    input_csv
):
    # """
    # 写入一行 (P, S) 到 CSV. 若某个为空则写空。
    # """
    dset = dataset_h5['data'][trace_name]
    # tr_name = dset.attrs["trace_name"]
    # station_name = dset.attrs["receiver_code"]
    # station_lat  = dset.attrs["receiver_latitude"]
    # station_lon  = dset.attrs["receiver_longitude"]
    # station_elv  = dset.attrs["receiver_elevation_m"]
    # net_code     = dset.attrs["network_code"]
    # parts = tr_name.split('_')
    # instrument_type = parts[2] if len(parts) > 2 else '??'

    # start_time_str = dset.attrs["trace_start_time"]
    # try:
    #     start_dt = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S.%f')
    # except:
    #     start_dt = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')


    df=pd.read_csv(input_csv)
    dataset=df[df['trace_name']==trace_name]

    trace_name = dataset["trace_name"].values
    station_name = dataset["station_code"].values
    station_lat = dataset["station_latitude_deg"].values
    station_lon = dataset["station_longitude_deg"].values
    station_elv = dataset["station_elevation_m"].values
    start_time = dataset["source_origin_time"].values
    network_name = dataset["station_code"].values
    instrument_type = "??"


    start_time= pd.to_datetime(start_time).strftime("%Y/%m/%d %H:%M:%S")

    start_time=pd.to_datetime(start_time.values)

    # 采样率假设 100 Hz
    p_time = (start_time + timedelta(seconds=p_idx/60.0)) if p_idx is not None else None
    s_time = (start_time + timedelta(seconds=s_idx/60.0)) if s_idx is not None else None

    event_start = p_time
    event_end   = s_time

    def _date_convertor(r):
        if r is None:
            return ""
        elif isinstance(r, datetime):
            return r.strftime('%Y-%m-%d %H:%M:%S.%f')
        elif isinstance(r, str):
            return r  # or parse it if needed
        else:
            return str(r)


    data_3c = np.array(dset, dtype=np.float32)
    # data_3c = np.array(dset, dtype=np.float32)
    p_snr = _get_snr(data_3c[:,0], p_idx, 200) if p_idx else None
    s_snr = _get_snr(data_3c[:,1], s_idx, 200) if s_idx else None

    csv_writer.writerow([
        trace_name,
        network_name,
        station_name,
        instrument_type,
        station_lat,
        station_lon,
        station_elv,
        event_start.strftime('%Y-%m-%d %H:%M:%S.%f') if not pd.isnull(event_start) else "",
        event_end.strftime('%Y-%m-%d %H:%M:%S.%f') if not pd.isnull(event_end) else "",
        p_time.strftime('%Y-%m-%d %H:%M:%S.%f') if not pd.isnull(p_time) else "",
        round(p_prob,3) if p_prob else "",
        "",  # p_uncertainty
        p_snr if p_snr else "",
        s_time.strftime('%Y-%m-%d %H:%M:%S.%f') if not pd.isnull(s_time) else "",
        round(s_prob,3) if s_prob else "",
        "",  # s_uncertainty
        s_snr if s_snr else ""
    ])



############################################
# 主函数：只输出每条 trace 最高的 P 及其匹配到的 S
############################################
class SeismicDataset(Dataset):
    def __init__(self, hdf5_file, trace_names, normalization_mode='std'):
        self.hdf5_file = hdf5_file
        self.trace_names = trace_names
        self.normalization_mode = normalization_mode
        self.h5 = h5py.File(hdf5_file, 'r')

    def __len__(self):
        return len(self.trace_names)

    def __getitem__(self, idx):
        trace_name = self.trace_names[idx]
        data = np.array(self.h5['data'][trace_name], dtype=np.float32)

        if self.normalization_mode == 'std':
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-10)
        elif self.normalization_mode == 'max':
            data = data / (np.max(np.abs(data), axis=0) + 1e-10)

        # data = data.T  # shape: (3, N)
        data = torch.from_numpy(data)
        return {
            'input': data, 
            'trace_name': trace_name
        }

    def __del__(self):
        self.h5.close()


def phasenet_prediction(
    input_csv, 
    input_hdf5, 
    store_dir,
    P_threshold=0.3,
    S_threshold=0.3,
    sample_ratio=0.1,
    random_seed=49,
    batch_size=64,
    device=None
):
    """
    预测后，每条 trace 只保留一个事件（最高P），并匹配首个S。
    """
    start_time = time.time()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1) Read CSV
    df_csv = pd.read_csv(input_csv, low_memory=False)
    trace_list = df_csv["trace_name"].tolist()

    # 2) Subsample
    random.seed(random_seed)
    total_samples = len(trace_list)
    sample_size = int(total_samples * sample_ratio)
    if sample_size < len(trace_list):
        trace_list = random.sample(trace_list, sample_size)

    # 3) Create Dataset
    dataset = SeismicDataset(input_hdf5, trace_list, normalization_mode='std')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=min(mp.cpu_count(), 8), pin_memory=True)

    # 4) Load PhaseNet
    model = sbm.PhaseNet.from_pretrained("stead").to(device)
    model.eval()

    all_p, all_s = [], []
    print("Starting inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            inputs = batch['input'].float().to(device)
            outputs = model(inputs)  # (B, 2, N)
            all_p.append(outputs[:, 0, :].cpu().numpy())
            all_s.append(outputs[:, 1, :].cpu().numpy())

    all_p = np.concatenate(all_p, axis=0)  # shape: (num_traces, N)
    all_s = np.concatenate(all_s, axis=0)

    np.savez("/home/yuan/code/SeisMoE/data/test/phasenet.npz",
             T=trace_list,
             P=all_p,
             S=all_s)

    if not os.path.exists(store_dir):
        os.makedirs(store_dir)

    csv_path = os.path.join(store_dir, f"phasenet_predictions_{random_seed}.csv")
    print(f"Writing results to: {csv_path}")

    num_events = 0  # 统计事件总数

    with open(csv_path, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow([
            'file_name', 'network', 'station', 'instrument_type',
            'station_lat', 'station_lon', 'station_elv',
            'event_start_time', 'event_end_time', 
            'p_arrival_time', 'p_probability', 'p_uncertainty', 'p_snr',
            's_arrival_time', 's_probability', 's_uncertainty', 's_snr'
        ])

        # 5) Post-processing: picking & writing
        with h5py.File(input_hdf5, 'r') as h5data:
            for i, trace_name in enumerate(tqdm(trace_list, desc="Post-processing")):
                p_array = all_p[i]
                s_array = all_s[i]

                # 只保留最高 P
                matches = phasenet_picker(P_threshold, S_threshold, p_array, s_array)
                # matches 字典里只会有 1 个键值对: {best_p_idx: (best_s_idx, best_p_prob, best_s_prob)}

                for p_idx, val in matches.items():
                    s_idx, p_prob, s_prob = val
                    phasenet_prediction_writer(
                        h5data, trace_name, p_idx, s_idx, p_prob, s_prob, writer,input_csv
                    )
                    num_events += 1
                fp.flush()

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n*** Finished PhaseNet prediction on {len(trace_list)} traces.")
    print(f"*** Total picks (events) detected: {num_events}")
    print(f"*** Elapsed: {elapsed:.1f} seconds (~{elapsed/60:.1f} min).")
    print(f"*** CSV saved to: {csv_path}")


# 只在脚本直接执行时运行下列示例
if __name__ == "__main__":
    input_csv = "DATA/metadata_no_noise.csv"
    input_hdf5 = "DATA/waveforms_all_nonoise.hdf5"
    store_dir = "/home/yuan/code/SeisMoE/data/test"
    phasenet_prediction(
        input_csv=input_csv,
        input_hdf5=input_hdf5,
        store_dir=store_dir,
        random_seed=36,
        P_threshold=0.1,   # 可以根据需要升高阈值
        S_threshold=0.1,
        sample_ratio=0.001,  
        batch_size=200,
        device=None
    )
