# SeisMoE

地震相位拾取（P/S picking）项目。当前主线是一套 **PyTorch + MoE** 的训练/评估
流水线，在 STEAD 上训练并与 PhaseNet、EQTransformer 对比；历史实验代码保留在
`src/`、`notebooks/`、`paper/`、`results/` 中。

## 1. 新流水线（推荐）

```text
seismoe/
├── data/
│   └── dataset.py       # pread 读波形 + 归一化 + 数据增强 + 高斯标签
├── models/
│   ├── moe.py           # top-k 稀疏 MoE + shared expert + load balancing
│   └── seismoe.py       # 残差 U-Net + bottleneck Transformer/MoE
├── train.py             # 训练、EMA、验证、按 picking F1 保存 best.pt
├── evaluate.py          # 单模型评估、阈值扫描、picks CSV
├── transfer.py          # TEXD/INSTANCE bucket 读取、微调 Dataset、迁移评估
└── eval.py              # 峰值检测与 TP/FP/FN/F1/MAE
scripts/
├── build_offsets.py     # 一次扫描生成 HDF5 字节偏移索引（只存元数据）
├── build_manifest.py    # 偏移 + 标签 + 按 source_id 的事件级划分
├── build_leakfree_manifest.py  # 只保留官方 STEAD test trace（泄漏检查）
├── final_compare.py     # SeisMoE / PhaseNet / EQTransformer 同划分、同 trace 对比
├── transfer_eval.py     # TEXD / INSTANCE 的零样本与微调后评估
└── label_onset_check.py # 检验 P 误差是否来自整秒标签（AIC 起跳点）
tests/                   # 模型/数据/评估的快速单测
```

### 1.1 数据读取：不复制、不移动 STEAD

`/mnt/f/STEAD/STEAD.hdf5` 有 126 万个 dataset，每个是 6000×3 float32 连续存储。
在 9p 挂载上按名字 `h5py` 查找约 120 ms/个，但原始 72 kB `pread` 只有约 0.6 ms。
`scripts/build_offsets.py` 把 HDF5 的 metadata cache 调到 128 MB，一次性解析出
**全部 1,265,657 个波形的字节偏移**（约 8 分钟），只保存偏移和标签，绝不把波形
数据复制到 WSL 本地盘。之后训练用 `os.pread` 直接读 `/mnt/f`，单线程约 1400 条/s。

```bash
python scripts/build_offsets.py     # -> data/stead_offsets.npy / stead_names.npy
python scripts/build_manifest.py    # -> data/stead_manifest.csv
```

划分：按 `source_id` 做 **event-level** 80/10/10 划分（同一地震的道不会跨集合），
noise 按 trace_name 哈希划分。当前规模：

| split | event | noise |
|---|---:|---:|
| train | 650,256 | 188,156 |
| val | 15,000 | 4,000 |
| test | 25,000 | 6,000 |

### 1.2 训练

```bash
conda activate dl
python -m seismoe.train \
  --data data/stead_manifest.csv \
  --out runs/moe_200k \
  --epochs 30 --batch-size 64 --workers 10 \
  --max-train 200000 --max-val 3000 \
  --channels 32,64,96,128,192,256 \
  --moe-layers 2 --experts 8 --top-k 2 --expert-hidden 512 \
  --lr 1e-3 --dropout 0.1 --ema 0.997
```

- 输入 3×6000（100 Hz，60 s），逐通道去均值/方差归一化。
- 数据增强：±3 s 随机平移、真实噪声按 SNR 混合、幅度缩放、极性翻转、通道
  dropout、高斯噪声。
- 标签：P/S 高斯（σ=10 采样点）+ noise 辅助通道。
- 损失：`BCE(P) + BCE(S) + 0.2·BCE(N) + 0.01·load_balance`。
- AdamW + warmup + cosine、bf16 autocast、梯度裁剪、权重 EMA。
- 验证集按 `P/S F1@0.2 s`（阈值 0.1）选 `best.pt`，同时记录 0.1/0.2/0.5 s。
  最终发布的 `runs/moe_650k_ft/final.pt` 取第 3 个 epoch 的 raw 权重（比 EMA
  best 在 val/test 上高 <0.002），二者都可直接用。

单卡 RTX 4060 Ti 8 GB：200k 样本约 7.5 min/epoch，显存约 4 GB。

### 1.3 评估

```bash
# 单模型：在 val 上扫阈值，再在 test 上用固定阈值
python -m seismoe.evaluate --model seismoe --ckpt runs/moe_650k_ft/final.pt \
  --split val --max-traces 5000 --sweep --out results/moe_val
python -m seismoe.evaluate --model seismoe --ckpt runs/moe_650k_ft/final.pt \
  --split test --thr-p 0.1 --thr-s 0.1 --out results/moe_test

# 与 PhaseNet / EQTransformer 同划分、同 trace 对比（各自用正确归一化）
python scripts/final_compare.py --ckpt runs/moe_650k_ft/final.pt --max-val 3000
```

阈值扫描：`0.02–0.7`（迁移数据用低阈值）；峰值检测：
`scipy.find_peaks(height=thr, distance=100)`，P 取最高峰，S 取 P 之后的最高峰。
TP 容差同时报告 0.1/0.2/0.5 s。

## 2. 模型

```text
3×6000
  └─ stem 7×1 conv, 32 ch
  └─ 6 级残差 encoder（32→64→96→128→192→256），max 32× 下采样到 188
  └─ 1 层 dense Transformer + 2 层 Sparse MoE Transformer（8 experts, top-2,
     shared expert, load-balancing loss）
  └─ 5 级 skip decoder（nearest up + conv + concat + residual）
  └─ 3×1 conv head → P / S / noise logits
```

总参数约 7.9 M（dense 变体约 2.65 M）。MoE 只放在 188-token 的 bottleneck：
此时每个 token 覆盖约 0.32 s，路由是“P 波列 / S 波列 / 噪声”的语义选择，而不是
局部纹理选择；计算量小、单卡稳定。`--moe-layers 0` 可直接训练同骨干的 dense
消融。

## 3. 结果（STEAD，event-level 划分）

完整 test（25,000 events + 6,000 noise），容差 0.5 s / 0.2 s：

| model | 预处理 | P F1@0.5 | S F1@0.5 | P F1@0.2 | S F1@0.2 |
|---|---|---:|---:|---:|---:|
| **SeisMoE** | ENZ, std | 0.986 | 0.966 | 0.953 | 0.896 |
| PhaseNet `original` | ENZ, std | 0.951 | 0.946 | 0.921 | 0.874 |
| PhaseNet `stead` | ZNE, max | 0.926 | 0.971 | 0.869 | 0.906 |
| EQTransformer `original` | ZNE, std | **0.990** | 0.973 | **0.986** | 0.905 |
| EQTransformer `stead` | ZNE, max | 0.987 | **0.975** | 0.965 | 0.896 |

SeisMoE 的 P 明显超过两个 PhaseNet；对在 STEAD 上训练过的 EQTransformer 仍差
0.4–0.9 个点 @0.5 s。详细分析见 [`RESULTS.md`](RESULTS.md)，其中两个关键点：

1. **STEAD 的 P 标签是整秒量化的**（94.5% 在整秒 ±0.05 s 内）。EQTransformer 的
   P 预测 97% 落在整秒网格上，SeisMoE raw 只有 78%；把 P 峰按标签约定校准后
   P F1@0.2 从 0.953 提到 **0.970**。做物理走时应用 raw pick，做 benchmark
   对比可用校准后的数。
2. 在官方 STEAD `test.npy` 子集（排除预训练权重见过的 trace）上排名基本不变，
   说明 EQT 的优势不是单纯 trace 泄漏。

在 **INSTANCE**（干净 event-level 划分）上，从 STEAD 微调的 SeisMoE 在 test 上
四项指标都超过 SeisBench 的 in-domain PhaseNet / EQTransformer：

| model | P F1@0.2 | S F1@0.2 | P F1@0.5 | S F1@0.5 |
|---|---:|---:|---:|---:|
| **SeisMoE (STEAD → INSTANCE)** | **0.965** | **0.930** | **0.973** | **0.971** |
| PhaseNet `instance` | 0.956 | 0.929 | 0.964 | 0.962 |
| EQTransformer `instance` | 0.961 | 0.924 | 0.971 | 0.960 |

## 4. 扩展到其他数据集（TEXD / INSTANCE）

TEXD 与 INSTANCE 的 HDF5 是 bucket 布局（`f['data'][bucket][index]`，`(3,T)`，
**ZNE** 顺序，100 Hz），与 STEAD 的逐 trace 布局不同。`seismoe/transfer.py`
负责读取、转成 ENZ、逐通道标准化，并把 INSTANCE 的 120 s 窗口裁成包含 P/S
的 60 s 窗口（5 s pre-P），不做错误的重采样。

```bash
# 零样本：直接在 dev 上扫阈值
python scripts/transfer_eval.py --dataset txed --split dev \
  --model seismoe --ckpt runs/moe_650k_ft/final.pt --norm std --sweep

# 在目标数据集 train 上微调，dev 选模型，test 只评估一次
python -m seismoe.train --transfer txed --out runs/txed_ft \
  --epochs 15 --batch-size 32 --lr 2e-4 --workers 6 \
  --channels 32,64,96,128,192,256 --moe-layers 2 --experts 8 \
  --top-k 2 --expert-hidden 512 --dropout 0.1 --ema 0.99 \
  --val-thr 0.05 --init runs/moe_650k_ft/final.pt
python scripts/transfer_eval.py --dataset txed --split test \
  --model seismoe --ckpt runs/txed_ft/best.pt --thr-p 0.2 --thr-s 0.1

# INSTANCE 同理，把 --transfer/--dataset 换成 instance；
# 注意 INSTANCE 列名是大写 trace_P/S_arrival_sample，已在 TRANSFER_SPECS 配好。
```

TEXD 官方 split 是 trace 级（同一地震会跨 train/test），所以 TEXD 数值偏乐观；
INSTANCE 的 train/dev/test 是 event-level 无交集的。在 INSTANCE test 上，
SeisMoE 微调后 P/S F1@0.5 = 0.973/0.971，优于 SeisBench 的 in-domain
PhaseNet `instance`（0.964/0.962）和 EQTransformer `instance`（0.971/0.960）。

注意：STEAD 的 P 标签有 77% 是整秒（100 采样点）量化的，因此 0.2 s 容差下
“预测物理到时”的模型会被“背下量化标签”的模型压过；详见 `RESULTS.md`。

## 5. 历史代码

- `src/`：早期 TensorFlow/PyTorch 实验脚本（PhaseNet/EQTransformer/MoE 推理、
  数据预处理、旧版 MoE 模型 `model_moe_gpt_PS2_CPU.py` 等）。
- `notebooks/`、`results/`、`paper/`：旧论文的结果、图和 notebook。
- `models/Model_Store/`：旧实验权重（仅 `best_model.pth` 纳入 git）。
- `benchmark/`：旧推理时间对比实验（已从 git 删除，历史中可恢复）。

## 6. 环境

`conda activate dl`：Python 3.10、PyTorch 2.9.1+cu128、numpy、pandas、scipy、
h5py、seisbench 0.11.7、obspy。无 TensorFlow 依赖。

## 7. 测试

```bash
python tests/run_all.py
```
（环境未装 pytest；`tests/` 同时兼容 pytest。）
