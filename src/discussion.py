import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from model_moe_gpt_PS2_CPU import IMG1

def normalize_data(data, mode='std'):
    data -= np.mean(data, axis=1, keepdims=True)
    if mode == 'std':
        std_val = np.std(data, axis=1, keepdims=True)
        std_val[std_val == 0] = 1.0
        data /= std_val
    elif mode == 'max':
        max_val = np.max(np.abs(data), axis=1, keepdims=True)
        max_val[max_val == 0] = 1.0
        data /= max_val
    else:
        raise ValueError(f"Unsupported norm_mode: {mode}")
    return data

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered[ch] = filtfilt(b, a, data[ch])
    return filtered





# Usage example
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hdf5_path = "/home/cloudlab/意大利数据集instancegm/Instance.hdf5"
trace_name = 'bucket693$256,:3,:12000'

from scipy.signal import resample,butter,filtfilt
group = trace_name.split('$')[0].strip()
index = int(trace_name.split('$')[1].split(',')[0].strip())
f=h5py.File(hdf5_path,'r')
dataset = f['data'].get(group)
raw_data = dataset[index, :, :].astype(np.float32)  # shape: (3, n_samples)
target_length=6000
# 重采样到 target_length
if raw_data.shape[1] != target_length:
    normalized = np.zeros((3, target_length), dtype=np.float32)
    for ch in range(3):
        normalized[ch] = resample(raw_data[ch], target_length)
else:
    normalized = raw_data.astype(np.float32)



# 滤波：1-45Hz 带通滤波
normalized = bandpass_filter(normalized, lowcut=1.0, highcut=45.0, fs=100.0)

# 标准化
normalized = normalize_data(normalized, mode='std')
data=torch.from_numpy(normalized.T)
data=data.unsqueeze(0)
input_signal = data.clone().to(device)

# Load model
model = IMG1(num_experts=4, num_experts_per_tok=4).to(device)







def normalize_data1(data, mode='std'):
    data -= np.mean(data, axis=1, keepdims=True)

    if mode == 'std':
        std_val = np.std(data, axis=1, keepdims=True)
        std_val[std_val == 0] = 1.0
        data /= std_val
    elif mode == 'max':
        max_val = np.max(np.abs(data), axis=1, keepdims=True)
        max_val[max_val == 0] = 1.0
        data /= max_val
    else:
        raise ValueError(f"Unsupported norm_mode: {mode}")
    return data

def normalize_data2(data, mode='max'):
    data -= np.mean(data, axis=0, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=0, keepdims=True)
        assert (max_data.shape[-1] == data.shape[-1])
        max_data[max_data == 0] = 1
        data /= max_data

    elif mode == 'std':
        std_data = np.std(data, axis=0, keepdims=True)
        assert (std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
    return data

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    对每个通道数据进行带通滤波。
    data: 形状为 (3, T)
    lowcut, highcut: 滤波器截止频率
    fs: 采样率
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered[ch] = filtfilt(b, a, data[ch])
    return filtered



# def collect_expert_weights(model, input_data):
model.eval()
p_weights = {}  # {layer_name: expert_weights}
s_weights = {}  # {layer_name: expert_weights}
with torch.no_grad():
    # Process with MultiScaleFeatureExtractor
    feat = model.feature_extractor(input_signal.permute(0, 2, 1))  # [batch, 64, seq_len]
    print(f"MultiScaleFeatureExtractor output shape: {feat.shape}")

    # 设置 feat_branch 为 [16, 64]
    # 直接定义 feat_branch 为 [1, 16, 16] 的张量
    feat_branch = torch.nn.Parameter(torch.randn(1, 16, 47)).to(feat.device)
    feat_branch=feat_branch
    print(f"feat_branch shape: {feat_branch.shape}")  # 应输出 torch.Size([1, 16, 16])


    # Temporary dimension reduction for all MoE layers
    feat_reduced = feat.permute(0, 2, 1)  # [batch, seq_len, 64]
    weight = torch.nn.Parameter(torch.randn(16,64)).to(feat.device)  # [in_features=64, out_features=16]
    feat_reduced = torch.nn.functional.linear(feat_reduced, weight)  # [batch, seq_len, 16]
    feat_reduced = feat_reduced.permute(0, 2, 1)  # [batch, 16, seq_len]
    print(f"Reduced feat shape for all MoE layers: {feat_reduced.shape}")

    if hasattr(model, 'context_dp0'):
        try:
            moe_out, moe_weights, router_logits = model.context_dp0(feat)
            print(f"context_dp0 router_logits shape: {router_logits.shape}")
            router_probs = torch.softmax(router_logits, dim=-1)[0].cpu().numpy()
            weights = router_probs.mean(axis=0) if router_probs.ndim == 2 else router_probs
            p_weights['context_dp0'] = weights
            s_weights['context_dp0'] = weights
            print(f"context_dp0 expert weights: {weights}")
        except Exception as e:
            print(f"Error processing context_dp0: {str(e)}")
    else:
        print("Warning: context_dp0 not found in model")

    # context_dp1
    if hasattr(model, 'context_dp1'):
        try:
            moe_out, moe_weights, router_logits = model.context_dp1(feat)
            print(f"context_dp1 router_logits shape: {router_logits.shape}")
            router_probs = torch.softmax(router_logits, dim=-1)[0].cpu().numpy()
            weights = router_probs.mean(axis=0) if router_probs.ndim == 2 else router_probs
            p_weights['context_dp1'] = weights
            s_weights['context_dp1'] = weights
            print(f"context_dp1 expert weights: {weights}")
        except Exception as e:
            print(f"Error processing context_dp1: {str(e)}")
    else:
        print("Warning: context_dp1 not found in model")

    # context_d0
    if hasattr(model, 'context_d0'):
        try:
            moe_out, moe_weights, router_logits = model.context_d0(feat_reduced)
            print(f"context_d0 router_logits shape: {router_logits.shape}")
            router_probs = torch.softmax(router_logits, dim=-1)[0].cpu().numpy()
            weights = router_probs.mean(axis=0) if router_probs.ndim == 2 else router_probs
            p_weights['context_d0'] = weights
            s_weights['context_d0'] = weights
            print(f"context_d0 expert weights: {weights}")
        except Exception as e:
            print(f"Error processing context_d0: {str(e)}")
    else:
        print("Warning: context_d0 not found in model")

    # context_d
    if hasattr(model, 'context_d'):
        try:
            moe_out, moe_weights, router_logits = model.context_d(feat_reduced)
            print(f"context_d router_logits shape: {router_logits.shape}")
            router_probs = torch.softmax(router_logits, dim=-1)[0].cpu().numpy()
            weights = router_probs.mean(axis=0) if router_probs.ndim == 2 else router_probs
            p_weights['context_d'] = weights
            s_weights['context_d'] = weights
            print(f"context_d expert weights: {weights}")
        except Exception as e:
            print(f"Error processing context_d: {str(e)}")
    else:
        print("Warning: context_d not found in model")

    # Process p_branch

    # 修改 p_branch 和 s_branch 的处理逻辑
    try:
        # p_branch 可能只返回 (output, router_logits)
        _,_, p_router_logits = model.p_branch.attn_branch(input_dim=16)(feat_branch)  # 改为接收2个值
        print(f"p_branch router_logits shape: {p_router_logits.shape}")
        p_router_probs = torch.softmax(p_router_logits, dim=-1)[0].cpu().numpy()
        p_weights['p_branch'] = p_router_probs.mean(axis=0) if p_router_probs.ndim == 2 else p_router_probs
        print(f"p_branch expert weights: {p_weights['p_branch']}")
    except Exception as e:
        print(f"Error processing p_branch: {str(e)}")

    try:
        # s_branch 同理
        _,_, s_router_logits = model.s_branch.attn_branch(feat_branch)
        print(f"s_branch router_logits shape: {s_router_logits.shape}")
        s_router_probs = torch.softmax(s_router_logits, dim=-1)[0].cpu().numpy()
        s_weights['s_branch'] = s_router_probs.mean(axis=0) if s_router_probs.ndim == 2 else s_router_probs
        print(f"s_branch expert weights: {s_weights['s_branch']}")
    except Exception as e:
        print(f"Error processing s_branch: {str(e)}")


            # return p_weights, s_weights




# Collect and visualize expert weights
# p_weights, s_weights = collect_expert_weights(model, input_signal)


import pandas as pd

# 将 expert 权重转换为 DataFrame
def weights_to_dataframe(weights_dict, label):
    df = pd.DataFrame.from_dict(weights_dict, orient='index')
    df.columns = [f'expert_{i}' for i in range(df.shape[1])]
    df.index.name = 'layer'
    df['branch'] = label
    return df.reset_index()

# 转换为 DataFrame
df_p = weights_to_dataframe(p_weights, 'P')
df_s = weights_to_dataframe(s_weights, 'S')

# 合并两个 DataFrame
df_instance = pd.concat([df_p, df_s], ignore_index=True)

# 打印 DataFrame 结果
df_instance
output_csv = Path(__file__).resolve().parents[1] / "data" / "expert_weights" / "expert_weights_instance.csv"
output_csv.parent.mkdir(parents=True, exist_ok=True)
df_instance.to_csv(output_csv, index=False)
