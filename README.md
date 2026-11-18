# SeisMoE

地震相位拾取项目：PhaseNet / EQTransformer / SeisMoE（MoE）在 STEAD、DiTing、Instance、TEXD、EastAfrican 等数据集上的训练、预测、对比实验与论文图。

## 目录结构

```text
SeisMoE/
├── src/                     # 全部 Python 代码（平铺，脚本之间直接 import）
│   ├── Auxiliary.py                    # 模型基础模块
│   ├── model_moe_gpt_PS2_CPU.py        # SeisMoE 模型定义（2 token）
│   ├── model_moe_gpt_PS3_CPU.py        # SeisMoE 模型定义（3 token）
│   ├── model_moe_gpt_PS_CPU_signal.py  # 信号分支模型变体
│   ├── seis_diting_*.py                # DiTing 训练/预测脚本
│   ├── seis_instance_*.py              # Instance 训练/预测脚本
│   ├── seis_stead_*.py                 # STEAD 训练/预测脚本
│   ├── seis_Texd_*.py                  # TEXD 训练/预测脚本
│   ├── seis_eastafican_*.py            # EastAfrican 训练/预测脚本
│   ├── seis_pre_phasenet_*.py          # STEAD PhaseNet 多种子/并行版本
│   ├── seis_pre_rqtransformer_*.py      # STEAD RQTransformer 多种子/并行版本
│   ├── predict_PS.py                   # 批量预测入口
│   ├── predict_subfunction_*.py        # 预测后处理子函数
│   ├── discussion.py                   # 专家权重提取
│   ├── tets.py                         # seisbench 标注接口的 PhaseNet 流程
│   └── preprocess/                     # HDF5 切分、Instance CSV 过滤
├── notebooks/               # 根目录结果 notebook（CPU 版 + 早期版本）
├── data/
│   ├── DiTing_Predict/                 # 各模型预测 CSV
│   ├── EastAfican_Predict/
│   ├── Instance_Predict/
│   ├── STEAD_Predict/
│   ├── TEXD_Predict/
│   ├── STEAD_PhaseNet/                 # PhaseNet 多种子预测结果
│   ├── model123_stead_predict/         # model1/2/3 的 STEAD 预测
│   ├── expert_weights/                 # 专家权重 CSV
│   ├── samples/                        # 样例数据
│   └── test/                           # 零散测试结果
├── results/
│   ├── disturb_photo/       # 扰动实验：notebook + 结果图 + 预测 CSV
│   ├── MOE_PLOT/            # MoE 可视化：notebook + Figure（PDF/SVG）
│   ├── RESULT_FIGURE/       # 论文结果图（按数据集分目录）
│   ├── RESULT_FIGURE.zip    # 2025-05 旧快照，含 12 张现目录中没有的图
│   └── paper_figures/       # 根目录散落的结果图（误差分布、示例波形）
├── models/
│   ├── best_model_CPU.pth   # 旧版最佳模型
│   └── Model_Store/         # 各实验权重；仅 best_model.pth 纳入 git
├── benchmark/               # 推理时间对比（原 Time_compare）
├── paper/                   # 模型结构图、专家分布图
├── scripts/                 # 运行脚本（run_models.sh）
├── logs/                    # 早期训练队列日志
└── README.md
```

## 代码说明

- `src/` 是平铺结构，因为实验脚本之间使用裸模块名互相导入，例如
  `from model_moe_gpt_PS2_CPU import IMG1`、`from Auxiliary import ...`。
  这样直接运行 `python src/seis_stead_Moe.py` 即可，不需要设置 `PYTHONPATH`。
- `MOE_PLOT/` 中原来重复的 `Auxiliary.py`、`model_moe_gpt_PS2_CPU.py` 未再复制，
  统一使用 `src/` 中的唯一副本。
- `benchmark/`（原 `Time_compare`）是一个自成体系的推理时间对比实验，保留了自己的
  “优化版”脚本、日志和结果。它通过两个符号链接引用 `src/` 中的共享模块：
  `benchmark/Auxiliary.py`、`benchmark/model_moe_gpt_PS2_CPU.py`。
  Windows 克隆后如符号链接未启用，可把这两个文件复制过去，或从仓库根目录用
  `PYTHONPATH=src python benchmark/...` 运行。
- 主要依赖：`torch`、`numpy`、`pandas`、`scipy`、`h5py`、`matplotlib`、`tqdm`、
  `seisbench`、`obspy`、`transformers`、`torchviz` / `torchview`、Jupyter。

## 数据与路径

- 仓库内的预测结果、专家权重、图件路径已统一改写为当前仓库的绝对路径
  （`/home/yuan/code/SeisMoE/...`），换机器后可用全局替换调整根路径。
- 训练/预测脚本中的**输入数据**仍保留原实验机器的绝对路径
  （如 `/media/cloudlab/...`、`/mnt/d/AI_Seismic_Data/...`、`/home/cloudlab/...`）。
  这些是外部 HDF5/CSV 数据集，不在仓库内，运行前需按本机数据位置修改。
- notebook 中的 `sys.path` 已改为 `/home/yuan/code/SeisMoE/src`，
  模型加载路径已改为 `models/Model_Store/...`。

## 模型权重

`models/Model_Store/` 共约 1.6 GB，包含 13 个 `best_model.pth`（约 96 MB）以及
每轮 epoch 权重和 checkpoint。当前 `.gitignore` 只保留 `best_model.pth`：

```text
models/Model_Store/**/*.pth
!models/Model_Store/**/best_model.pth
```

其余权重仍保留在本地磁盘，只是不进入 git。若以后需要版本化全部权重，可改用
Git LFS（注意 GitHub 免费 LFS 配额为 1 GB）。

## 未纳入仓库的内容

- `test/seisbench/`：seisbench 官方仓库的完整克隆（含独立 `.git`，约 115 MB），
  作为第三方依赖排除；需要时用 pip 安装或单独克隆。
- `__pycache__/`、`.vscode/`、空目录 `temp/`、`seisbench_cloudlab/`。
