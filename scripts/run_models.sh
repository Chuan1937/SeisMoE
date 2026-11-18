#!/bin/bash

# 定义基础目录
BASE_DIR="/home/yuan/code/SeisMoE/src"
TRAIN_DIR="/home/cloudlab/下载/DEEP_Pytorch/Pytorch_STEAD"

# 切换到第一个目录并运行第一个模型
cd "$BASE_DIR" || exit
echo "======================================"
echo "Running seis_stead_Moe.py"
echo "======================================"
python seis_stead_Moe.py

# 运行第二个模型（同一目录）
echo "======================================"
echo "Running seis_stead_PhaseNet.py"
echo "======================================"
python seis_stead_PhaseNet.py

# 运行第三个模型（同一目录）
echo "======================================"
echo "Running seis_stead_Eqtransformer.py"
echo "======================================"
python seis_stead_Eqtransformer.py

# 切换到训练目录并运行训练脚本
cd "$TRAIN_DIR" || exit
echo "======================================"
echo "Running train_preD_moe_PS_STEAD_CPU.py"
echo "======================================"
python train_preD_moe_PS_STEAD_CPU.py

echo "======================================"
echo "All models executed successfully!"
echo "======================================"