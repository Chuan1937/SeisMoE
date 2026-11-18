#!/bin/bash
# 运行所有12个优化脚本

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
echo "运行所有优化脚本"
echo "========================================"

for script in "${SCRIPTS[@]}"; do
    echo ""
    echo "运行: $script"
    python "$script"
    echo "完成: $script"
done

echo ""
echo "========================================"
echo "收集结果..."
echo "========================================"
python collect_results.py

echo ""
echo "所有脚本运行完成！"