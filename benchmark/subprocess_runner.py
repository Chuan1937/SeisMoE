#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
子进程运行器 - 逐个隔离运行12个优化脚本，收集时间
"""

import subprocess
import os
import sys
import json
import glob
import re

SCRIPT_DIR = "/home/yuan/code/SeisMoE/benchmark"

# 12个目标脚本（按数据集分组）
CONFIGS = [
    # Txed 数据集 — 3个模型
    ("Txed", "Eqtransformer", "seis_Texd_Eqtransformer.py"),
    ("Txed", "PhaseNet",      "seis_Texd_phasenet.py"),
    ("Txed", "MoE",           "seis_Texd_Moe.py"),

    # Instance 数据集 — 3个模型
    ("Instance", "Eqtransformer", "seis_Instance_Eqtransformer.py"),
    ("Instance", "PhaseNet",      "seis_Instance_PhaseNet.py"),
    ("Instance", "MoE",           "seis_instance_Moe.py"),

    # DiTing 数据集 — 3个模型
    ("DiTing", "Eqtransformer", "seis_diting_Eqtransformer.py"),
    ("DiTing", "PhaseNet",      "seis_diting_PhaseNet.py"),
    ("DiTing", "MoE",           "seis_diting_Moe.py"),

    # STEAD 数据集 — 3个模型
    ("STEAD", "Eqtransformer", "seis_stead_Eqtransformer.py"),
    ("STEAD", "PhaseNet",      "seis_stead_PhaseNet.py"),
    ("STEAD", "MoE",           "seis_stead_Moe.py"),
]


def run_one(script_name, dataset, model):
    """运行单个脚本并返回耗时"""
    script_path = os.path.join(SCRIPT_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"  [SKIP] 脚本不存在: {script_path}")
        return None

    print(f"\n{'='*55}")
    print(f"  {dataset} / {model}")
    print(f"  脚本: {script_name}")
    print(f"{'='*55}")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=1800,  # 30分钟超时
        )

        print(result.stdout)
        if result.stderr:
            # 过滤 DtypeWarning
            stderr_lines = [l for l in result.stderr.split('\n')
                           if 'DtypeWarning' not in l and l.strip()]
            if stderr_lines:
                print("STDERR:", '\n'.join(stderr_lines[:5]))

        # 从stdout提取时间（新格式: "检出: N  时间: X.XX 秒"）
        time_match = re.search(r'时间:\s*([\d.]+)\s*秒', result.stdout)
        if time_match:
            elapsed = float(time_match.group(1))
            return elapsed

        # 旧格式: "运行时间: X.XX 秒"
        time_match = re.search(r'运行时间:\s*([\d.]+)\s*秒', result.stdout)
        if time_match:
            elapsed = float(time_match.group(1))
            return elapsed

        # 旧格式: "time=X.XX"
        time_match = re.search(r'time=\s*([\d.]+)', result.stdout)
        if time_match:
            elapsed = float(time_match.group(1))
            return elapsed

        # 回退: 读取 time.txt
        time_file = os.path.join(SCRIPT_DIR, f"{dataset}_{model}", "time.txt")
        if os.path.exists(time_file):
            with open(time_file) as f:
                try:
                    return float(f.read().strip())
                except ValueError:
                    pass

        print("  [WARN] 无法提取运行时间")
        return None

    except subprocess.TimeoutExpired:
        print("  [TIMEOUT] 超过30分钟")
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def generate_markdown(results):
    datasets = ['Txed', 'Instance', 'DiTing', 'STEAD']
    models = ['Eqtransformer', 'PhaseNet', 'MoE']
    labels = {'Txed': 'Txed', 'Instance': 'Instance', 'DiTing': 'DiTing', 'STEAD': 'STEAD'}

    md = "# 地震波形拾取模型运行时间对比\n\n"
    md += "## 测试环境\n"
    md += "- **CPU**: 24核 | **内存**: 64GB | **GPU**: 8GB显存\n"
    md += "- **数据量**: 每个数据集10,000条波形\n"
    md += "- **批处理大小**: 1024~2048（根据模型自动调整）\n\n"
    md += "## 运行时间（秒）\n\n"
    md += "| 数据集 | EQTransformer | PhaseNet | MoE |\n"
    md += "|--------|:-----------:|:--------:|:---:|\n"

    for ds in datasets:
        row = f"| {labels[ds]} |"
        for m in models:
            key = (ds, m)
            val = results.get(key)
            if val is not None:
                row += f" {val:.2f} |"
            else:
                row += " N/A |"
        md += row + "\n"

    md += "\n## 优化说明\n"
    md += "- HDF5文件句柄保持打开，避免重复IO\n"
    md += "- 增大批处理大小至1024~2048\n"
    md += "- 优化DataLoader参数: num_workers=4, prefetch_factor=2\n"
    md += "- GPU同步计时: torch.cuda.synchronize()\n"
    md += "- 每个测试完成后清除GPU缓存\n"
    return md


def main():
    results = {}
    for dataset, model, script in CONFIGS:
        elapsed = run_one(script, dataset, model)
        results[(dataset, model)] = elapsed
        # 短暂停顿，让GPU资源释放
        import time
        time.sleep(2)

    # 打印摘要
    summary = generate_markdown(results)
    print("\n" + "=" * 55)
    print(summary)

    # 保存
    with open(os.path.join(SCRIPT_DIR, 'time_results.json'), 'w') as f:
        serializable = {f"{k[0]}_{k[1]}": v for k, v in results.items()}
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    with open(os.path.join(SCRIPT_DIR, 'README.md'), 'w') as f:
        f.write(summary)

    print(f"\n结果已保存到: {SCRIPT_DIR}/time_results.json 和 README.md")


if __name__ == "__main__":
    main()
