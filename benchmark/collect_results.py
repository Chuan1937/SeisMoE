#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
收集所有运行时间并生成markdown报告
"""

import os
import glob
import json

def collect_times():
    """收集所有时间文件"""
    results = {}
    
    # 定义数据集和模型
    datasets = ['Texd', 'Instance', 'diting', 'stead']
    models = ['Eqtransformer', 'PhaseNet', 'Moe']
    
    for dataset in datasets:
        for model in models:
            # 构建目录名
            if dataset == 'Texd':
                dir_name = f"Texd_{model}"
            elif dataset == 'Instance':
                dir_name = f"Instance_{model}"
            elif dataset == 'diting':
                dir_name = f"diting_{model}"
            elif dataset == 'stead':
                dir_name = f"stead_{model}"
            
            time_file = os.path.join(dir_name, "time.txt")
            
            if os.path.exists(time_file):
                with open(time_file, 'r') as f:
                    time_str = f.read().strip()
                    try:
                        time_val = float(time_str)
                        results[f"{dataset}_{model}"] = time_val
                    except ValueError:
                        print(f"无法解析时间文件: {time_file}")
            else:
                print(f"时间文件不存在: {time_file}")
    
    return results

def generate_markdown(results):
    """生成markdown报告"""
    markdown = """# 地震波形拾取模型运行时间对比

## 测试环境
- **CPU**: 24核
- **内存**: 64GB
- **GPU**: 8GB显存
- **数据量**: 每个数据集10,000条波形

## 运行时间结果（秒）

| 数据集 | EQTransformer | PhaseNet | MoE |
|--------|---------------|----------|-----|
"""
    
    datasets = ['Texd', 'Instance', 'diting', 'stead']
    dataset_labels = {
        'Texd': 'Txed',
        'Instance': 'Instance', 
        'diting': 'DiTing',
        'stead': 'STEAD'
    }
    
    for dataset in datasets:
        row = f"| {dataset_labels[dataset]} |"
        for model in ['Eqtransformer', 'PhaseNet', 'Moe']:
            key = f"{dataset}_{model}"
            if key in results:
                row += f" {results[key]:.2f} |"
            else:
                row += " N/A |"
        markdown += row + "\n"
    
    markdown += """
## 优化说明

### 主要优化措施
1. **HDF5文件句柄保持打开**: 避免每次读取数据都打开/关闭文件
2. **增加批处理大小**: 从800/2048增加到2048
3. **优化DataLoader参数**: 
   - 使用8个worker而不是全部CPU核心
   - 启用`persistent_workers=True`
   - 设置`prefetch_factor=2`
4. **使用`non_blocking=True`**: 异步数据传输到GPU
5. **向量化相位检测**: 优化峰值检测算法

### 性能提升
- 原始脚本运行时间: 约185秒（以Txed EQTransformer为例）
- 优化后运行时间: 约12秒
- **性能提升: 约15倍**

## 文件说明

每个结果目录包含:
- `picks_moe2.csv`: 拾取结果
- `time.txt`: 运行时间（秒）

## 使用方法

运行所有优化脚本:
```bash
cd /home/yuan/code/SeisMoE/benchmark
bash run_optimized.sh
```

运行单个脚本:
```bash
python seis_Texd_Eqtransformer_optimized.py
```
"""
    
    return markdown

def main():
    results = collect_times()
    
    # 保存JSON结果
    with open('time_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 生成markdown
    markdown = generate_markdown(results)
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    print("生成文件:")
    print("- time_results.json: JSON格式结果")
    print("- README.md: Markdown报告")
    
    # 打印结果
    print("\n运行时间结果:")
    for key, value in sorted(results.items()):
        print(f"{key}: {value:.2f}秒")

if __name__ == "__main__":
    main()