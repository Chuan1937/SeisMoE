import os
import time
import csv
import h5py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
import seisbench.models as sbm
import sys
import gc,random
sys.path.append(r'/home/yuan/code/SeisMoE/src')
from predict_subfunction_DPS import _detect_peaks,prediction_writer,picker,_get_snr
# ------- 自定义数据加载类 -------
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

        data = data.T  # EQTransformer 需要 [3, 6000]
        data = torch.from_numpy(data)
        return {'input': data, 'trace_name': trace_name}

    def __del__(self):
        self.h5.close()


# ------- 主预测函数 -------
def eqt_prediction(input_csv, input_hdf5, store_dir, detection_threshold=0.5, P_threshold=0.3, S_threshold=0.3, batch_size=20, device=None,random_seed=42,ratio=0.1):
    start_time = time.time()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}')
    model = sbm.EQTransformer.from_pretrained("stead").to(device)
    model.eval()

    df = pd.read_csv(input_csv)
    predict_list = df["trace_name"].tolist()

    random.seed(random_seed)
    df_csv = pd.read_csv(input_csv, low_memory=False)  # 解决 DtypeWarning
    predict_list = df_csv.trace_name.tolist()
    total_samples = len(predict_list)
    sample_size = int(total_samples*ratio)
    predict_list = random.sample(predict_list, sample_size)

    dataset = SeismicDataset(input_hdf5, predict_list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=mp.cpu_count())

    all_det, all_p, all_s = [], [], []
    print("Start predicting...")
    with torch.no_grad():
        for batch in tqdm(loader):
            inputs = batch['input'].to(device).float()
            outputs = model(inputs)
            all_det.append(outputs[0].cpu().numpy())
            all_p.append(outputs[1].cpu().numpy())
            all_s.append(outputs[2].cpu().numpy())

    all_det = np.concatenate(all_det, axis=0)
    all_p = np.concatenate(all_p, axis=0)
    all_s = np.concatenate(all_s, axis=0)

    # 保存 .npz 格式预测数据
    # npz_save_path = os.path.join(store_dir, "eqt_predictions.npz")
    # np.savez(npz_save_path,
    #          D=all_det,
    #          P=all_p,
    #          S=all_s,
    #          T=predict_list)
    # print(f'*** Saved prediction arrays to {npz_save_path}')

    prob_dic = {
        'DD_mean': all_det, 'PP_mean': all_p, 'SS_mean': all_s,
        'DD_std': np.zeros_like(all_det), 'PP_std': np.zeros_like(all_p), 'SS_std': np.zeros_like(all_s),
    }

    del all_det, all_p, all_s
    gc.collect()

    if not os.path.exists(store_dir):
        os.makedirs(store_dir)

    csv_path = os.path.join(store_dir, f'Prediction_results_eqtransformer_{random_seed}.csv')
    with open(csv_path, 'w') as csv_file:
        predict_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        predict_writer.writerow(['file_name', 'network', 'station', 'instrument_type',
                                 'station_lat', 'station_lon', 'station_elv',
                                 'event_start_time', 'event_end_time', 'detection_probability',
                                 'detection_uncertainty', 'p_arrival_time', 'p_probability',
                                 'p_uncertainty', 'p_snr', 's_arrival_time', 's_probability',
                                 's_uncertainty', 's_snr'])

        detection_memory = []
        with h5py.File(input_hdf5, 'r') as dataset_h5:
            for i, trace_name in enumerate(tqdm(predict_list)):
                # print(f"Processing trace {i}: {trace_name}")
                dat = np.array(dataset_h5['data'][trace_name])
                dat = (dat - np.mean(dat, axis=0)) / (np.std(dat, axis=0) + 1e-10)
                matches, _, _ = picker(detection_threshold, P_threshold, S_threshold,
                                       prob_dic['DD_mean'][i], prob_dic['PP_mean'][i], prob_dic['SS_mean'][i],
                                       prob_dic['DD_std'][i], prob_dic['PP_std'][i], prob_dic['SS_std'][i])

                if (len(matches) >= 1) and (matches[list(matches)[0]][3] and matches[list(matches)[0]][6]):
                    if (matches[list(matches)[0]][6] - matches[list(matches)[0]][3]) < 60 * 100:
                        snr = [_get_snr(dat, matches[list(matches)[0]][3], window=100),
                               _get_snr(dat, matches[list(matches)[0]][6], window=100)]
                        dataset_trace = dataset_h5['data'][trace_name]
                        detection_memory = prediction_writer(dataset_trace, predict_writer, csv_file, matches, snr, detection_memory)

    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f'\n*** Finished prediction in {int(hours)}h {int(minutes)}m {round(seconds, 2)}s.')
    print(f'*** Detected: {len(detection_memory)} events.')
    print(f'*** Results saved in: "{store_dir}/Prediction_results.csv"')
    sys.exit(0)


# -------- 示例调用 --------
if __name__ == "__main__":
    input_csv = "/mnt/nn/Code/DATA/STEAD/STEAD.csv"
    input_hdf5 = "/mnt/nn/Code/DATA/STEAD/STEAD.hdf5"
    store_dir = "/home/yuan/code/SeisMoE/data/STEAD_Predict/STEAD_EQTransformer_Predict"

    eqt_prediction(
        input_csv=input_csv,
        input_hdf5=input_hdf5,
        store_dir=store_dir,
        detection_threshold=0.5,
        P_threshold=0.3,
        S_threshold=0.3,
        batch_size=200,
        device=None,
        random_seed=49,
        ratio=0.1
    )

