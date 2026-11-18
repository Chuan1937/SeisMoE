import csv
import math
import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm
from model_moe_gpt_PS3_CPU import IMG1  # 请确保该模块和模型定义正确
import gc
import sys
import multiprocessing as mp
from queue import Empty
import random
# 峰值检测函数
def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    dx = x[1:] - x[:-1]
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
    if ind.size and indnan.size:
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    if ind.size and threshold > 0:
        dx_local = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx_local < threshold)[0])
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0
        ind = np.sort(ind[~idel])
    return ind

# P、S波选取函数
def picker(P_threshold, S_threshold, yh2, yh3, spLimit=60):
    pp_arr = _detect_peaks(yh2, mph=P_threshold, mpd=1)
    ss_arr = _detect_peaks(yh3, mph=S_threshold, mpd=1)
    P_PICKS = {pick: [np.round(yh2[pick], 3), None] for pick in pp_arr}
    S_PICKS = {pick: [np.round(yh3[pick], 3), None] for pick in ss_arr}
    matches = {}
    available_S = list(S_PICKS.keys())
    for p_time in sorted(P_PICKS.keys()):
        candidate_Ss = [s for s in available_S if p_time < s < p_time + spLimit * 100]
        if candidate_Ss:
            s_time = max(candidate_Ss, key=lambda x: S_PICKS[x][0])
            matches[p_time] = [
                s_time, 1.0, None,
                p_time, P_PICKS[p_time][0], P_PICKS[p_time][1],
                s_time, S_PICKS[s_time][0], S_PICKS[s_time][1]
            ]
            available_S.remove(s_time)
    return matches, {}, yh3

# 计算信噪比
def _get_snr(data, pat, window=200):
    snr = None
    if pat is not None:
        try:
            pat = int(pat)
            if pat >= window and (pat + window) < len(data):
                nw1 = data[pat - window: pat]
                sw1 = data[pat: pat + window]
            elif pat < window and (pat + window) < len(data):
                window = pat
                nw1 = data[pat - window: pat]
                sw1 = data[pat: pat + window]
            elif (pat + window) > len(data):
                window = len(data) - pat
                nw1 = data[pat - window: pat]
                sw1 = data[pat: pat + window]
            snr = round(10 * math.log10((np.percentile(sw1, 95) / np.percentile(nw1, 95)) ** 2), 1)
        except Exception:
            pass
    return snr

# 写入预测结果到CSV
def prediction_writer(dataset, predict_writer, csvPr, matches, snr, detection_memory):
    trace_name = dataset.attrs["trace_name"]
    station_name = "{:<4}".format(dataset.attrs["receiver_code"])
    station_lat = dataset.attrs["receiver_latitude"]
    station_lon = dataset.attrs["receiver_longitude"]
    station_elv = dataset.attrs["receiver_elevation_m"]
    start_time = dataset.attrs["trace_start_time"]
    network_name = "{:<2}".format(dataset.attrs["network_code"])
    instrument_type = "{:<2}".format(trace_name.split('_')[2])
    try:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')
    except Exception:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')

    def _date_convertor(r):
        if isinstance(r, str):
            parts = r.split('.')
            if len(parts) == 1:
                new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S')
            else:
                new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S.%f')
        else:
            new_t = r
        return new_t

    for match, match_value in matches.items():
        ev_strt = start_time + timedelta(seconds=match / 100)
        ev_end = start_time + timedelta(seconds=match_value[0] / 100)
        doublet = [st for st in detection_memory if abs((st - ev_strt).total_seconds()) < 2]
        if not doublet:
            det_prob = round(match_value[1], 2)
            det_unc = match_value[2]
            p_time = start_time + timedelta(seconds=match_value[3] / 100) if match_value[3] is not None else None
            p_prob = match_value[4]
            p_unc = round(match_value[5], 2) if match_value[5] is not None else None
            s_time = start_time + timedelta(seconds=match_value[6] / 100) if match_value[6] is not None else None
            s_prob = match_value[7]
            s_unc = round(match_value[8], 2) if match_value[8] is not None else None
            predict_writer.writerow([
                trace_name, network_name, station_name, instrument_type,
                station_lat, station_lon, station_elv,
                _date_convertor(ev_strt), _date_convertor(ev_end),
                det_prob, det_unc,
                _date_convertor(p_time) if p_time is not None else None, p_prob, p_unc, snr[0],
                _date_convertor(s_time) if s_time is not None else None, s_prob, s_unc, snr[1]
            ])
            csvPr.flush()
            detection_memory.append(ev_strt)
    return detection_memory


import csv
import math
import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm
from model_moe_gpt_PS3_CPU import IMG1
import gc
import sys
import multiprocessing as mp
import random
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置多进程启动方法为 'spawn'
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

# 数据集类
class SeismicDataset(Dataset):
    def __init__(self, hdf5_file, trace_names, normalization_mode='std'):
        self.hdf5_file = hdf5_file
        self.trace_names = trace_names
        self.normalization_mode = normalization_mode

    def __len__(self):
        return len(self.trace_names)

    def __getitem__(self, idx):
        trace_name = self.trace_names[idx]
        with h5py.File(self.hdf5_file, 'r') as h5:
            data = np.array(h5['data'][trace_name], dtype=np.float32)
        if self.normalization_mode == 'std':
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-10)
        elif self.normalization_mode == 'max':
            data = data / (np.max(np.abs(data), axis=0) + 1e-10)
        data = torch.from_numpy(data)
        return {'input': data, 'trace_name': trace_name}

# 峰值检测、P/S波选取、信噪比计算、结果写入函数（保持不变）
# (_detect_peaks, picker, _get_snr, prediction_writer) ...

# 修改后的 CPU 单线程筛选函数（带进度条）
def process_predictions_single_thread(hdf5_file, batch_results, P_threshold, S_threshold, spLimit, output_csv_path, detection_memory):
    with open(output_csv_path, 'a', newline='') as csvPr_gen:
        predict_writer = csv.writer(csvPr_gen, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        with h5py.File(hdf5_file, 'r') as h5:
            # 将进度条嵌入循环
            for trace_name, p_prob, s_prob in tqdm(batch_results, desc="筛选处理中", unit="样本"):
                dataset_h5 = h5['data'][trace_name]
                data_array = np.array(dataset_h5)
                matches, _, _ = picker(P_threshold, S_threshold, p_prob, s_prob, spLimit=spLimit)
                if matches:
                    for match_key, match_value in matches.items():
                        snr = [_get_snr(data_array, match_value[3], window=100),
                               _get_snr(data_array, match_value[6], window=100)]
                        detection_memory = prediction_writer(dataset_h5, predict_writer, csvPr_gen,
                                                            {match_key: match_value}, snr, detection_memory)
    return detection_memory

# 主预测函数
def predict(input_csv, input_hdf5, store_dir, model_path, P_threshold, S_threshold, batch_size=1500, spLimit=60,random_seed=99):
    start_time_overall = time.time()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError("GPU不可用，程序终止。")
    print(f'使用设备: {device}')
    
    # 在主进程中加载模型
    model = IMG1().to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    except Exception:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    os.makedirs(store_dir, exist_ok=True)
    

    df_csv = pd.read_csv(input_csv, low_memory=False)
    # predict_list = df_csv.trace_name.tolist()
    # total_samples = len(predict_list)
    # sample_size = int(total_samples * 0.1)
    # predict_list = random.sample(predict_list, sample_size)

    predict_list = df_csv['trace_name'].sample(frac=0.1, random_state=36).tolist()

    dataset = SeismicDataset(input_hdf5, predict_list)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    output_csv_path = os.path.join(store_dir, 'Prediction_results_{random_seed}.csv')
    with open(output_csv_path, 'w', newline='') as csvPr_gen:
        predict_writer = csv.writer(csvPr_gen, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        predict_writer.writerow(['file_name', 'network', 'station', 'instrument_type',
                                 'station_lat', 'station_lon', 'station_elv',
                                 'event_start_time', 'event_end_time', 'detection_probability',
                                 'detection_uncertainty', 'p_arrival_time', 'p_probability',
                                 'p_uncertainty', 'p_snr', 's_arrival_time', 's_probability',
                                 's_uncertainty', 's_snr'])

    # 主进程进行预测
    print("开始批次预测（主进程）...")
    batch_results = []
    pbar_predict = tqdm(data_loader, desc="批次预测中", unit="批次")
    for batch_idx, batch in enumerate(pbar_predict):
        inputs = batch['input'].to(device)
        trace_names_batch = batch['trace_name']

        with torch.no_grad():
            PS, _ = model(inputs)
            p_prob_batch = PS[0].cpu().numpy()
            s_prob_batch = PS[1].cpu().numpy()

        for i in range(len(trace_names_batch)):
            batch_results.append((trace_names_batch[i], p_prob_batch[i], s_prob_batch[i]))

        # 释放 GPU 内存
        del inputs, PS, p_prob_batch, s_prob_batch
        torch.cuda.empty_cache()

        if batch_idx % 10 == 0:
            gc.collect()

        pbar_predict.set_postfix({'已处理': min((batch_idx + 1) * batch_size, len(dataset)), '总数': len(dataset)})

    # 单线程筛选阶段
    print("开始结果筛选（单线程）...")
    detection_memory = []
    detection_memory = process_predictions_single_thread(
        input_hdf5, batch_results, P_threshold, S_threshold, spLimit, output_csv_path, detection_memory
    )

    end_time_overall = time.time()
    delta = end_time_overall - start_time_overall
    hour = int(delta / 3600)
    minute = int((delta - hour * 3600) / 60)
    seconds = delta - hour * 3600 - minute * 60

    dd = pd.read_csv(output_csv_path, low_memory=False)
    print('\n')
    print('*** 预测完成，用时: {} 小时, {} 分钟, {:.2f} 秒。'.format(hour, minute, seconds))
    print('*** 检测到的事件数: {}。'.format(len(dd)))
    print('*** 结果保存至 --> "{}"'.format(store_dir))
    sys.exit(0)

if __name__ == "__main__":
    input_csv = "/media/cloudlab/aea7978e-7b9d-408e-9287-70f6f6d38d4a/Code/DATA/STEAD/STEAD.csv"
    input_hdf5 = "/media/cloudlab/aea7978e-7b9d-408e-9287-70f6f6d38d4a/Code/DATA/STEAD/STEAD.hdf5"
    store_dir = r"/home/yuan/code/SeisMoE/data/STEAD_Predict/STEAD_Moe"
    model_path = r'/home/yuan/code/SeisMoE/models/Model_Store/CPU3/best_model.pth'
    P_threshold = 0.1
    S_threshold = 0.1
    random_seed=36
    predict(input_csv, input_hdf5, store_dir,model_path, P_threshold, S_threshold,
            batch_size=1500, spLimit=60,random_seed=random_seed)
