#!/bin/bash
# 运行所有优化版本的脚本

set -e

SCRIPTS=(
    "seis_Texd_Eqtransformer_optimized.py"
    "seis_Texd_phasenet_optimized.py"
    "seis_Texd_Moe_optimized.py"
    "seis_Instance_Eqtransformer_optimized.py"
    "seis_Instance_PhaseNet_optimized.py"
    "seis_instance_Moe_optimized.py"
    "seis_diting_Eqtransformer_optimized.py"
    "seis_diting_PhaseNet_optimized.py"
    "seis_diting_Moe_optimized.py"
    "seis_stead_Eqtransformer_optimized.py"
    "seis_stead_PhaseNet_optimized.py"
    "seis_stead_Moe_optimized.py"
)

echo "========================================"
echo "开始运行所有优化脚本"
echo "系统配置: 64GB内存, 8GB显存"
echo "========================================"

for script in "${SCRIPTS[@]}"; do
    echo ""
    echo "========================================"
    echo "运行: $script"
    echo "开始时间: $(date)"
    echo "========================================"
    
    python "$script"
    
    echo "完成: $script"
    echo "结束时间: $(date)"
done

echo ""
echo "========================================"
echo "所有脚本运行完成"
echo "========================================"