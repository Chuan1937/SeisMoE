#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修改12个时间对比脚本，统一数据量、数据路径和输出目录。
"""

import os
import re
import glob
import shutil

# 定义数据集映射
datasets = {
    'txed': {
        'input_csv': '/mnt/d/AI_Seismic_Data/txed/Texd.csv',
        'hdf5_path': '/mnt/d/AI_Seismic_Data/txed/Texd.hdf5',
    },
    'Instance': {
        'input_csv': '/mnt/d/AI_Seismic_Data/Instance/INSTANCE.csv',
        'hdf5_path': '/mnt/d/AI_Seismic_Data/Instance/INSTANCE.hdf5',
    },
    'stead': {
        'input_csv': '/mnt/d/AI_Seismic_Data/STEAD/STEAD.csv',
        'hdf5_path': '/mnt/d/AI_Seismic_Data/STEAD/STEAD.hdf5',
    },
    'diting': {
        'input_csv': '/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.csv',
        'hdf5_path': '/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.hdf5',
    }
}

# 固定样本数量
FIXED_SAMPLES = 10000

# 输出目录
OUTPUT_DIR = '/home/yuan/code/SeisMoE/benchmark'

# 原始脚本目录
SCRIPT_DIR = '/home/yuan/code/SeisMoE/src'

# 获取所有12个脚本
scripts = glob.glob(os.path.join(SCRIPT_DIR, 'seis_*_*.py'))
# 过滤出目标脚本
target_scripts = []
for script in scripts:
    basename = os.path.basename(script)
    # 检查是否匹配目标模式
    if any(keyword in basename for keyword in ['Texd', 'Instance', 'stead', 'diting']):
        # 排除一些非目标脚本（如 eastafican）
        if 'eastafican' not in basename:
            target_scripts.append(script)

print(f"找到 {len(target_scripts)} 个目标脚本:")
for s in target_scripts:
    print(f"  {os.path.basename(s)}")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 修改每个脚本
for script_path in target_scripts:
    basename = os.path.basename(script_path)
    # 确定数据集
    if 'Texd' in basename:
        dataset_key = 'txed'
    elif 'Instance' in basename or 'instance' in basename:
        dataset_key = 'Instance'
    elif 'stead' in basename:
        dataset_key = 'stead'
    elif 'diting' in basename:
        dataset_key = 'diting'
    else:
        print(f"跳过无法识别数据集的脚本: {basename}")
        continue
    
    # 读取脚本内容
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. 替换 input_csv 路径
    # 查找 input_csv 参数赋值
    pattern = r"(input_csv\s*=\s*)(['\"])([^'\"]+)\2"
    replacement = f"\\1\\2{datasets[dataset_key]['input_csv']}\\2"
    content = re.sub(pattern, replacement, content)
    
    # 2. 替换 hdf5_path 路径
    pattern = r"(hdf5_path\s*=\s*)(['\"])([^'\"]+)\2"
    replacement = f"\\1\\2{datasets[dataset_key]['hdf5_path']}\\2"
    content = re.sub(pattern, replacement, content)
    
    # 3. 替换 output_dir 路径
    # 创建子目录：Time_compare/<dataset>_<model>/
    model_name = basename.replace('seis_', '').replace('.py', '')
    output_subdir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_subdir, exist_ok=True)
    
    pattern = r"(output_dir\s*=\s*)(['\"])([^'\"]+)\2"
    replacement = f"\\1\\2{output_subdir}\\2"
    content = re.sub(pattern, replacement, content)
    
    # 4. 替换采样比例为固定数量
    # 查找 frac=... 模式并替换为 n=FIXED_SAMPLES
    # 同时确保有 random_state=36
    pattern = r"\.sample\(frac=([^,)]+)(,\s*random_state=\d+)?\)"
    replacement = f".sample(n={FIXED_SAMPLES}, random_state=36)"
    content = re.sub(pattern, replacement, content)
    
    # 5. 确保 CUDA_VISIBLE_DEVICES 设置为 "0"
    # 查找 os.environ["CUDA_VISIBLE_DEVICES"] 并设置为 "0"
    pattern = r'(os\.environ\["CUDA_VISIBLE_DEVICES"\]\s*=\s*)(["\'])[^"\']*\2'
    replacement = r'\g<1>"0"'
    content = re.sub(pattern, replacement, content)
    
    # 6. 添加计时代码和写入时间文件
    # 在 if __name__ == "__main__": 块中添加计时
    # 首先检查是否已经有计时代码
    if "import time" not in content:
        # 在导入部分添加 time
        content = content.replace("import os", "import os\nimport time")
    
    # 在 main 函数调用前后添加计时
    # 查找 if __name__ == "__main__": 块
    pattern = r'(if\s+__name__\s*==\s*["\']__main__["\']\s*:)'
    # 我们需要在该块中添加计时代码
    # 简单方法：在块的开头添加 start = time.time()，在结尾添加 end = time.time() 并写入文件
    
    # 由于每个脚本的结构略有不同，我们采用更通用的方法：
    # 在文件末尾添加一个计时包装器
    # 但这样可能会破坏原有结构。最好修改现有的 main 调用。
    
    # 查找 main 函数调用
    pattern = r'(main\()'
    # 我们可以在 main 调用前后添加计时代码
    # 但更简单的方法是：在脚本末尾添加一个计时函数，并修改 main 调用
    
    # 让我们检查脚本是否已经有计时代码
    if "time.time()" in content and "time=" in content:
        # 已经有计时代码，但可能需要调整输出
        # 将 print(f"time={end-start}") 改为写入文件
        pattern = r'print\(f"time=\{end-start\}"\)'
        time_file = os.path.join(output_subdir, 'time.txt')
        replacement = f'with open("{time_file}", "w") as tf:\n            tf.write(str(end-start))\n        print(f"time={{end-start}}")'
        content = re.sub(pattern, replacement, content)
    else:
        # 没有计时代码，需要添加
        # 在 if __name__ == "__main__": 块中添加
        # 找到该块的开始位置
        main_block_match = re.search(r'(if\s+__name__\s*==\s*["\']__main__["\']\s*:)', content)
        if main_block_match:
            # 在该行后添加计时代码
            insert_pos = main_block_match.end()
            # 获取该行后的缩进
            line_start = content.rfind('\n', 0, main_block_match.start()) + 1
            line = content[line_start:main_block_match.end()]
            indent = line[:len(line) - len(line.lstrip())]
            
            # 添加计时代码
            time_code = f"\n{indent}import time\n{indent}start = time.time()"
            content = content[:insert_pos] + time_code + content[insert_pos:]
            
            # 在文件末尾添加结束计时和写入文件
            time_file = os.path.join(output_subdir, 'time.txt')
            end_code = f"\n{indent}end = time.time()\n{indent}with open('{time_file}', 'w') as tf:\n{indent}    tf.write(str(end-start))\n{indent}print(f'time={{end-start}}')"
            content = content.rstrip() + end_code
    
    # 写入修改后的脚本
    output_script_path = os.path.join(OUTPUT_DIR, basename)
    with open(output_script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已修改并保存: {output_script_path}")

print("\n所有脚本修改完成。")