import os
import math
import h5py
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
# ========= 配置区域 =========
CSV_PATH = "/home/cloudlab/Code_DL/DEEP_STEAD_PYTORCH/Pytorch_STEAD/DATA/metadata.csv"
HDF5_PATH = "/home/cloudlab/Code_DL/DEEP_STEAD_PYTORCH/Pytorch_STEAD/DATA/waveforms.hdf5"
TMP_DIR   = "/home/cloudlab/Code_DL/DEEP_STEAD_PYTORCH/Pytorch_STEAD/DATA/tmp_hdf5"      # 存放临时文件的目录
N_WORKERS = mp.cpu_count()  # 并行进程数，你也可写成: os.cpu_count()

# 给最终合并的文件起一个名字（后面再合并）
MERGED_HDF5_PATH = "/home/cloudlab/Code_DL/DEEP_STEAD_PYTORCH/Pytorch_STEAD/DATA/pick_waveforms_merged.hdf5"

def read_and_write_sublist(trace_sublist, subset_idx):
    """
    每个进程执行的函数：读取 trace_sublist 中的所有数据，
    并写到一个临时 HDF5 文件里 (只属于本进程)。
    """
    # 先打开原始 HDF5（只读）
    with h5py.File(HDF5_PATH, 'r') as fin:
        data_group_in = fin['data']  # 假设里面是 "data" 分组
        
        # 临时文件名，比如 tmp_dir/part_0.hdf5, tmp_dir/part_1.hdf5
        tmp_file_path = os.path.join(TMP_DIR, f"part_{subset_idx}.hdf5")
        
        # 写模式打开临时文件
        with h5py.File(tmp_file_path, 'w') as fout:
            group_out = fout.create_group('data')
            
            # 遍历本进程负责的 trace_name 列表
            for trace_name in trace_sublist:
                # 解析 trace_name
                a = trace_name.split('$')[0].strip()
                b = trace_name.split('$')[1]
                c = b.split(',')[0].strip()
                
                # 读取数据
                data_in = data_group_in[a][int(c), :, :]
                
                # 写入临时文件
                group_out.create_dataset(
                    trace_name,
                    data=data_in,
                    compression='gzip',     # 可选
                    compression_opts=9
                )
    
    return tmp_file_path  # 返回临时文件的路径

def main():
    # ========== Step A: 将 CSV 里的 trace_name 切分 ==========
    df = pd.read_csv(CSV_PATH)
    trace_names = df['trace_name'].dropna().tolist()
    
    # 创建临时目录
    os.makedirs(TMP_DIR, exist_ok=True)
    
    # 切分 trace_names，到 N_WORKERS 份
    chunk_size = math.ceil(len(trace_names) / N_WORKERS)
    sublists = [
        trace_names[i:i+chunk_size]
        for i in range(0, len(trace_names), chunk_size)
    ]
    
    # ========== Step B: 并行处理每个 sublist，生成临时文件 ==========
    tmp_files = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {}
        # 提交任务
        for idx, sublist in enumerate(sublists):
            fut = executor.submit(read_and_write_sublist, sublist, idx)
            futures[fut] = idx
        
        # 用 tqdm 跟踪进度
        for fut in tqdm(as_completed(futures), total=len(futures)):
            tmp_file_path = fut.result()
            tmp_files.append(tmp_file_path)

    print("所有进程处理完成！产生了临时文件：")
    for p in tmp_files:
        print("  ", p)
    
    # ========== Step C: 合并临时文件到最终 HDF5 (单进程即可) ==========
    merge_hdf5(tmp_files, MERGED_HDF5_PATH)
    print("合并完毕！最终文件：", MERGED_HDF5_PATH)

def merge_hdf5(tmp_file_list, merged_path):
    """
    将若干个临时 HDF5 文件合并到一个目标文件中
    （每个临时文件里都有 data 分组和里面的若干键）。
    """
    with h5py.File(merged_path, 'w') as fout:
        group_out = fout.create_group('data')
        
        # 遍历所有临时文件
        for tmp_path in tmp_file_list:
            with h5py.File(tmp_path, 'r') as fin:
                group_in = fin['data']
                for ds_name in group_in.keys():
                    data_in = group_in[ds_name][:]
                    group_out.create_dataset(
                        ds_name,
                        data=data_in,
                        compression='gzip',
                        compression_opts=9
                    )


main()
