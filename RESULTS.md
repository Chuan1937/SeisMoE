# SeisMoE 实验结果（STEAD + TEXD/INSTANCE）

## 1. 协议

- 数据：`/mnt/f/STEAD/STEAD.hdf5`，**不复制、不移动**；训练用 `os.pread` 直读。
- 划分：按 `source_id` 分事件，train/val/test 内同一地震的道不跨集合；noise 按
  trace_name 哈希划分。
  - train: 650,256 events + 188,156 noise
  - val: 15,000 events + 4,000 noise
  - test: 25,000 events + 6,000 noise
- 输入：3×6000（100 Hz，60 s），逐通道去均值 / 标准差。
- 峰值检测：`find_peaks(height=thr, distance=100)`；P=最高峰，S=P 之后最高峰。
- 阈值：在 val 上扫，test 用固定阈值；指标为 TP/FP/FN/F1/MAE，容差
  0.1 / 0.2 / 0.5 s。
- baseline 预处理（在 val 上确认，错误顺序/归一化会掉 0.05–0.5 F1）：
  - PhaseNet `original`：ENZ + per-trace std
  - PhaseNet `stead`：ZNE + per-trace max
  - EQTransformer `original`：ZNE + per-trace std
  - EQTransformer `stead`：ZNE + per-trace max

## 2. 完整 test 结果（31,000 条：25,000 events + 6,000 noise）

模型：`runs/moe_650k_ft/final.pt`（650k 全量微调第 3 个 epoch 的 raw 权重；
比 EMA 权重在 val/test 上略好）。

| model | thr P/S | P F1@0.1 | S F1@0.1 | P F1@0.2 | S F1@0.2 | P F1@0.5 | S F1@0.5 | P MAE | S MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **SeisMoE** | 0.1/0.1 | **0.9000** | **0.8078** | **0.9527** | 0.8955 | 0.9858 | 0.9663 | 0.0354 | **0.0590** |
| PhaseNet `original` | 0.6/0.3 | 0.8443 | 0.7579 | 0.9207 | 0.8737 | 0.9508 | 0.9461 | 0.0537 | 0.0740 |
| PhaseNet `stead` | 0.4/0.4 | 0.8121 | 0.8075 | 0.8689 | **0.9056** | 0.9263 | 0.9709 | 0.0458 | 0.0604 |
| EQTransformer `original` | 0.2/0.2 | 0.9774 | 0.8060 | 0.9856 | 0.9053 | **0.9896** | 0.9727 | **0.0098** | 0.0611 |
| EQTransformer `stead` | 0.2/0.2 | 0.9167 | 0.7711 | 0.9652 | 0.8957 | 0.9868 | **0.9747** | 0.0433 | 0.0721 |

- SeisMoE 在 P 上明显超过两个 PhaseNet；S 超过 PhaseNet `original`。
- 对 EQTransformer（**在 STEAD 上训练过**）仍差 0.1–0.9 个点 @0.5 s；P@0.2
  的差距主要来自 P 标签量化（见第 4 节），不是漏检。
- 噪声鲁棒性（6,000 条 noise 上的误报率）：SeisMoE **0.22%**，
  EQTransformer 0.08% / 0.47%，PhaseNet `original` **16.1%**，
  PhaseNet `stead` 1.05%。

## 3. 官方 STEAD test 子集（排除预训练权重见过的 trace）

SeisBench 的 STEAD 划分来自 EQTransformer 仓库的
[`test.npy`](https://github.com/smousavi05/EQTransformer/raw/master/ModelsAndSampleData/test.npy)
（126,566 条）。运行 `python scripts/build_leakfree_manifest.py` 后，把我们的
val/test 与这个官方 test 集合取交集，得到 1,925 val + 3,036 test 条
（2,460 events + 576 noise）。这些 trace 不在 EQTransformer / PhaseNet-stead
的训练集里，因此可以检验“预训练模型只是背过 STEAD”这一假设。

| model | P F1@0.2 | S F1@0.2 | P F1@0.5 | S F1@0.5 |
|---|---:|---:|---:|---:|
| **SeisMoE** | 0.9466 | 0.8888 | 0.9852 | 0.9673 |
| PhaseNet `original` | 0.9205 | 0.8677 | 0.9521 | 0.9457 |
| PhaseNet `stead` | 0.8722 | 0.8979 | 0.9285 | 0.9708 |
| EQTransformer `original` | **0.9878** | 0.8986 | **0.9911** | 0.9746 |
| EQTransformer `stead` | 0.9602 | **0.8969** | 0.9874 | **0.9767** |

结论：EQTransformer 的优势不是（只）来自 trace 泄漏；它在未见过的官方 test
trace 上仍然很强。但它的 P 输出高度“贴合”STEAD 的整秒标签，见下节。

## 4. 关键发现：STEAD 的 P 标签是整秒量化的

- **77.3%** 的 P 标签恰好是 100 的倍数（整秒），**94.5%** 在整秒 ±5 个采样点
  （±0.05 s）内；S 标签没有这个现象（只有 1.3% 是整秒）。
- EQTransformer 的 P 预测有 **97.1%** 落在整秒 ±5 点内；SeisMoE raw 预测只有
  **78.1%**。也就是说 EQT 学会了把 P 峰“吸附”到标签的整秒网格上，而 SeisMoE
  更多是在预测波形上的物理起跳点。
- 把 SeisMoE 的 P 峰按已知标签约定吸到最近的整秒（用峰附近概率选择上/下网格）：
  val P@0.2 从 **0.953 → 0.971**，P@0.1 从 **0.902 → 0.957**，P@0.5 基本不变
  （0.986 → 0.987）。test 上 P@0.2 从 **0.951 → 0.970**。
- 对 val 上 975 个 10–50 点误差样本（97.7% 是整秒标签）做 AIC 起跳点检查：
  SeisMoE 有 **67.5%** 的样本比标签更接近波形起跳点（平均 |pick−label|=20.9 点，
  |pick−onset|=20.3 点）；EQTransformer 则 |pick−label|=4.1 点、
  |pick−onset|=26.9 点，只有 8.5% 更接近起跳点。这直接说明 EQT 的 P 峰在贴标签
  网格，而 SeisMoE 的 raw pick 更接近物理起跳。

因此本仓库同时报告两种 P 指标：

| 变体 | P F1@0.1 | P F1@0.2 | P F1@0.5 |
|---|---:|---:|---:|
| SeisMoE raw（物理起跳） | 0.9000 | 0.9527 | **0.9858** |
| SeisMoE + 整秒校准 | **0.9553** | **0.9709** | 0.9853 |
| EQTransformer `original` | 0.9774 | 0.9856 | 0.9896 |
| EQTransformer `stead` | 0.9167 | 0.9652 | 0.9868 |

整秒校准可用 `python scripts/snap_picks.py results/final/picks_SeisMoE.csv`
复现。它只应在“与 STEAD 标签协议对齐”的评测中使用；做物理到时（如走时反演）
时应使用 raw pick。

## 5. SeisMoE 训练轨迹

### 5.1 200k 子集，从零训练（30 epoch 计划，实际看收敛）

| epoch | train loss | val loss | P F1@0.5 | S F1@0.5 | P F1@0.2 | S F1@0.2 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0617 | 0.0139 | 0.9593 | 0.9401 | 0.9092 | 0.8610 |
| 1 | 0.0352 | 0.0119 | 0.9734 | 0.9503 | 0.9263 | 0.8779 |
| 2 | 0.0334 | 0.0110 | 0.9800 | 0.9586 | 0.9384 | 0.8832 |
| 3 | 0.0323 | 0.0105 | 0.9801 | 0.9599 | 0.9410 | 0.8870 |
| 4 | 0.0317 | 0.0104 | 0.9823 | 0.9609 | 0.9436 | 0.8905 |
| 5 | 0.0314 | 0.0109 | 0.9817 | 0.9652 | 0.9427 | 0.8965 |

### 5.2 650k 全量微调（从 200k best 初始化，lr 3e-4，5 epoch 计划）

| epoch | train loss | val loss | P F1@0.5 | S F1@0.5 | P F1@0.2 | S F1@0.2 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.03036 | 0.01023 | 0.9868 | 0.9664 | 0.9510 | 0.8960 |
| 1 | 0.03011 | 0.01001 | 0.9874 | 0.9667 | 0.9520 | 0.9004 |
| 2 (EMA best) | 0.02993 | 0.01006 | 0.9878 | 0.9683 | 0.9540 | 0.9044 |
| 3 (raw final) | 0.02985 | 0.01003 | 0.9876 | 0.9695 | 0.9534 | 0.9044 |

第 4–5 epoch 的边际收益 <0.001，提前停止。

## 6. 扩展到 TEXD / INSTANCE

两个数据集都是 bucket 布局、`(3,T)`、ZNE、100 Hz；`seismoe/transfer.py` 负责
转 ENZ、逐通道标准化、INSTANCE 的 120 s→60 s 窗口裁剪（5 s pre-P），**不重采样**。

### 6.1 TEXD

官方 split 是 trace 级（同一 `source_origin_time` 会同时出现在 train/dev/test：
train-dev 725、train-test 1,408 个重合 origin），所以下面的 TEXD 数字偏乐观，
只能作为“迁移是否 work”的 sanity check。

dev（1,022 events，无 noise，阈值在 dev 上扫）：

| model | 权重 | P F1@0.2 | S F1@0.2 | P F1@0.5 | S F1@0.5 |
|---|---|---:|---:|---:|---:|
| SeisMoE zero-shot | STEAD 650k | 0.6714 | 0.6576 | 0.7144 | 0.7037 |
| SeisMoE fine-tuned | + TEXD train | **0.7781** | **0.8182** | **0.8345** | **0.8881** |
| PhaseNet `original` zero-shot | NCEDC | 0.7265 | 0.7301 | 0.7756 | 0.7785 |
| PhaseNet `stead` zero-shot | STEAD | 0.6682 | 0.7463 | 0.7152 | 0.8038 |
| EQTransformer `original` zero-shot | STEAD | 0.5837 | 0.6332 | 0.6538 | 0.6906 |
| EQTransformer `stead` zero-shot | STEAD | 0.6659 | 0.6765 | 0.7319 | 0.7494 |

TEXD test（3,277 events，阈值用 dev 的 0.2/0.1）：

| model | P F1@0.1 | S F1@0.1 | P F1@0.2 | S F1@0.2 | P F1@0.5 | S F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|
| SeisMoE fine-tuned | 0.7132 | 0.7238 | 0.7751 | 0.8013 | 0.8377 | 0.8710 |

微调后 TEXD dev 的 P/S F1@0.5 从 0.714/0.704 提到 **0.835/0.888**，
超过所有 zero-shot baseline；但受 trace 级 split 影响，数值偏乐观。

### 6.2 INSTANCE（event-level 划分干净）

INSTANCE 的 train/dev/test 之间 `source_id` 完全无交集（1458/237/748 个事件），
因此这是干净的跨数据集验证。SeisBench 提供在 INSTANCE 上训练的 PhaseNet /
EQTransformer 权重，作为 in-domain baseline；SeisMoE 从 STEAD 650k 初始化后在
INSTANCE train 上微调 15 epoch，阈值在 dev 上选。

test（1,433 events，无 noise）：

| model | 训练数据 | P F1@0.2 | S F1@0.2 | P F1@0.5 | S F1@0.5 |
|---|---|---:|---:|---:|---:|
| **SeisMoE** | STEAD → INSTANCE train | **0.9653** | **0.9302** | **0.9730** | **0.9714** |
| PhaseNet `instance` | INSTANCE train | 0.9559 | 0.9294 | 0.9636 | 0.9623 |
| EQTransformer `instance` | INSTANCE train | 0.9613 | 0.9241 | 0.9710 | 0.9601 |

在 INSTANCE 上 SeisMoE 四项指标都优于两个 in-domain 预训练 baseline。

## 7. 四数据集联合模型（一个模型）

训练：从 STEAD 650k 的 `final.pt` 初始化，在 STEAD + TEXD + INSTANCE + DiTing
2.0 的混合数据上训练 2 epoch × 600k 样本；域采样权重
`STEAD 0.45 / DiTing 0.45 / TEXD 0.05 / INSTANCE 0.05`。统一预处理：ZNE→ENZ、
1–45 Hz 带通、60 s 窗口、逐通道 std。DiTing 的 130 GB HDF5 **没有拷贝**，只用
`os.pread` 读，训练峰值显存 4.2 GB、WSL 常驻内存 <8 GB。

test 结果（每域独立在 val 上选阈值）：

| dataset | model | P F1@0.2 | S F1@0.2 | P F1@0.5 | S F1@0.5 |
|---|---|---:|---:|---:|---:|
| **STEAD** (31k) | 联合模型（STEAD-routed） | 0.9489 | 0.8880 | 0.9838 | 0.9617 |
| | STEAD-only `final.pt` | 0.9527 | 0.8955 | **0.9858** | **0.9663** |
| | PhaseNet `original` | 0.9207 | 0.8737 | 0.9508 | 0.9461 |
| | PhaseNet `stead` | 0.8689 | **0.9056** | 0.9263 | 0.9709 |
| | EQTransformer `original` | **0.9856** | 0.9053 | **0.9896** | 0.9727 |
| | EQTransformer `stead` | 0.9652 | 0.8957 | 0.9868 | **0.9747** |
| **DiTing 2.0** (15k) | **联合模型** | **0.6442** | **0.6309** | **0.7614** | **0.6990** |
| | PhaseNet `diting` (in-domain, 50 Hz) | 0.5453 | 0.5236 | 0.5958 | 0.5972 |
| | PhaseNet `stead` | 0.5824 | 0.5106 | 0.6182 | 0.5777 |
| | EQTransformer `original` | 0.5649 | 0.4652 | 0.5691 | 0.5292 |
| | EQTransformer `stead` | 0.5425 | 0.4776 | 0.5797 | 0.5500 |
| **INSTANCE** (1,433) | **联合模型** | **0.9717** | **0.9347** | **0.9808** | **0.9717** |
| | PhaseNet `instance` (in-domain) | 0.9616 | 0.9281 | 0.9693 | 0.9616 |
| | EQTransformer `instance` (in-domain) | 0.9658 | 0.9251 | 0.9728 | 0.9590 |
| **TEXD** (3,277) | **联合模型** | 0.7297 | 0.7711 | **0.7839** | **0.8352** |
| | PhaseNet `original` (zero-shot, dev) | 0.7265 | 0.7301 | 0.7756 | 0.7785 |
| | PhaseNet `stead` (zero-shot, dev) | 0.6682 | 0.7463 | 0.7152 | 0.8038 |
| | EQTransformer `stead` (zero-shot, dev) | 0.6659 | 0.6765 | 0.7319 | 0.7494 |

- **DiTing / INSTANCE / TEXD**：联合模型在 P/S 的 0.2 s 和 0.5 s 指标上全面超过
  PhaseNet 和 EQTransformer（INSTANCE/DiTing 还是 in-domain 权重）。
- **STEAD**：联合模型的 P 超过两个 PhaseNet；S 超过 PhaseNet `original`，但低于
  PhaseNet `stead` 的 0.9709。对在 STEAD 上训练的 EQTransformer 仍低 0.6–1.3
  个点 @0.5 s；专用 STEAD 模型 `final.pt` 更接近（差 0.4–0.6 点）。
- `runs/stead_head/best.pt` 在联合模型上冻结共享 backbone，只加训一个 1.03M 的
  STEAD 专用 decoder（`--stead-head`）：STEAD P/S@0.5 从 0.9768/0.9562 提到
  **0.9838/0.9617**，其他三个数据集走原 decoder，结果不变。
- 进一步解冻 last encoder + context（150k STEAD）没有提升；与 STEAD specialist
  做概率 ensemble 也只到 P/S@0.5 ≈0.987/0.966，仍低于 EQT 的 0.990/0.973。
  STEAD 的剩余差距不是靠简单 adapter/ensemble 能补的。
- TEXD 官方 split 是 trace 级，数值偏乐观；DiTing/INSTANCE 是 event-level 干净划分。

## 8. 结论与局限

- 在严格 event-level STEAD 划分上，SeisMoE 明显优于两个 PhaseNet 变体；对在
  STEAD 上训练过的 EQTransformer 仍差 0.4–0.9 个点 @0.5 s，P@0.2 的差距主要
  由整秒标签造成（第 4 节）。
- 在官方 STEAD test 子集上排名不变，说明差距不是单纯的 trace 泄漏。
- 在 INSTANCE 的干净 event-level 划分上，SeisMoE 微调后四项指标都优于
  in-domain 的 PhaseNet/EQTransformer 预训练权重。
- 四数据集联合模型（一个模型）在 **DiTing 2.0、INSTANCE、TEXD** 上 P/S 的
  0.2 s 与 0.5 s 指标全面超过 PhaseNet / EQTransformer（INSTANCE 和 DiTing
  还是 in-domain 权重）。代价是 STEAD 比专用模型低约 1 个点：P 超过两个
  PhaseNet，S 超过 PhaseNet `original` 但低于 `stead`；比在 STEAD 上训练的
  EQTransformer 低 1.3–1.6 点 @0.5 s。
- 局限：STEAD 的 P 标签量化使 0.1/0.2 s 指标不能完全反映物理到时精度；TEXD
  官方 split 是 trace 级，数值偏乐观；DiTing 1.0（50 Hz）尚未用于联合训练。
- 下一步：给模型加 domain embedding/adapter，减少 STEAD 与 DiTing 的相互干扰；
  训练第二个 seed 做 ensemble；在 event-level 重划 TEXD 后重训；把 DiTing 1.0
  也纳入统一 100 Hz pipeline。

## 9. 复现

```bash
python scripts/build_offsets.py
python scripts/build_manifest.py
python -m seismoe.train ...                 # 见 README
python scripts/final_compare.py --data data/stead_manifest.csv \
  --ckpt runs/moe_650k_ft/final.pt --max-val 3000
# leak-free 子集
python scripts/build_leakfree_manifest.py
python scripts/final_compare.py --data data/stead_manifest_leakfree.csv \
  --ckpt runs/moe_650k_ft/final.pt --max-val 0 --max-test 0
# 四数据集联合模型
python scripts/build_diting_index.py --train 200000 --val 5000 --test 15000
python scripts/build_multidomain_manifest.py
python -m seismoe.train --multi --multi-manifest data/multidomain_manifest.csv \
  --out runs/multidomain --epochs 2 --epoch-size 600000 --batch-size 64 --workers 20 \
  --lr 1e-4 --ema 0.997 --domain-weights stead=0.45,diting=0.45,txed=0.05,instance=0.05 \
  --val-per-dataset 500 --init runs/moe_650k_ft/final.pt
python scripts/eval_multidomain.py --ckpt runs/multidomain/best.pt --max-val 1000
```
