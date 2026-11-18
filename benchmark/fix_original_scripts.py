#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键修复 Time_compare 中的脚本：保持原数据逻辑，加上优化
"""

import os
import re
import glob

SCRIPT_DIR = "/home/yuan/code/SeisMoE/benchmark"

# 需要修复的文件列表
TARGETS = [
    # Txed
    "seis_Texd_Eqtransformer.py",
    "seis_Texd_phasenet.py",
    "seis_Texd_Moe.py",
    # Instance
    "seis_Instance_Eqtransformer.py",
    "seis_instance_Moe.py",
    "seis_Instance_PhaseNet.py",
    # DiTing
    "seis_diting_Moe.py",
    "seis_diting_PhaseNet.py",
    "seis_diting_Eqtransformer.py",
    # STEAD
    "seis_stead_Eqtransformer.py",
    "seis_stead_Moe.py",
    "seis_stead_PhaseNet.py",
]

# 针对每个文件的具体修复
FILE_PATCHES = {}

# ---- 通用修复：HDF5文件句柄保持打开 ----
def fix_hdf5_open(content):
    """修复 __getitem__ 中的 HDF5 打开方式，改为保持句柄打开"""
    # 找到 __getitem__ 中 with h5py.File(self.hdf5_path, 'r') as f: 并替换
    old = "with h5py.File(self.hdf5_path, 'r') as f:"
    new = "if self.h5_file is None:\n            self.h5_file = h5py.File(self.hdf5_path, 'r')\n        f = self.h5_file"
    if old in content:
        content = content.replace(old, new)
    return content

def fix_init_h5file(content):
    """在 __init__ 中添加 self.h5_file = None"""
    # 找到 def __init__(self, ... 函数体，在最后添加
    # 简单方法：在 __init__ 的最后一行之后加
    lines = content.split('\n')
    in_init = False
    in_class_init = False
    init_end = -1
    for i, line in enumerate(lines):
        if 'def __init__' in line and 'self' in line:
            in_class_init = True
        if in_class_init:
            stripped = line.strip()
            if stripped.startswith('self.') or stripped.startswith('#'):
                init_end = i
            else:
                # 找到 def __init__ 的结束（下一个同缩进的 def 或空行后的不同缩进）
                pass

    # 简化：在 __init__ 末尾 self.sample_rate 行后插入
    if 'self.sample_rate = sample_rate' in content:
        content = content.replace(
            'self.sample_rate = sample_rate',
            'self.sample_rate = sample_rate\n        self.h5_file = None  # 延迟打开'
        )
    return content

def fix_dataloader(content):
    """优化 DataLoader 参数"""
    # 减少 num_workers
    content = re.sub(r'num_workers=mp\.cpu_count\(\)', 'num_workers=4', content)
    # 添加 prefetch_factor
    if 'prefetch_factor' not in content and 'pin_memory=True' in content:
        content = content.replace(
            'pin_memory=True\n',
            'pin_memory=True,\n        prefetch_factor=2\n'
        )
    content = content.replace('pin_memory=True', 'pin_memory=True, persistent_workers=False')
    return content

def fix_timing(content):
    """修复计时代码"""
    # 确保 import time 存在
    if 'import time\n' not in content and 'import time\n' not in content.replace('from', 'FROM'):
        # 在 import os 后添加
        content = content.replace('import os\n', 'import os\nimport time\n')
    
    # 修复 if __name__ 块的缩进
    # 查找 if __name__ == "__main__": 并确保后续行缩进正确
    pattern = r'(if\s+__name__\s*==\s*["\']__main__["\']\s*:)\n(import time\n)?(start\s*=\s*time\.time\()'
    if re.search(pattern, content):
        # 已经有时序代码，只需检查缩进
        pass
    else:
        # 添加 timing
        pass
    
    return content

def fix_moe_import(content):
    """修复 MoE 脚本的导入路径和模型路径"""
    # sys.path.append
    if 'sys.path.append' in content and '/home/cloudlab' in content:
        content = content.replace(
            'sys.path.append("/home/yuan/code/SeisMoE/src")',
            'sys.path.append("/home/yuan/code/SeisMoE/src")'
        )
    # 模型权重路径
    if 'load_state_dict' in content:
        # 保持不变，这是用户指定的路径
        pass
    return content

def fix_cuda(content):
    """修复 CUDA_VISIBLE_DEVICES"""
    content = re.sub(
        r'os\.environ\["CUDA_VISIBLE_DEVICES"\]\s*=\s*["\']\d+["\']',
        'os.environ["CUDA_VISIBLE_DEVICES"] = "0"',
        content
    )
    return content

def fix_sample(content):
    """确保采样数量统一为 10000"""
    content = re.sub(r'\.sample\(frac=[\d.]+\)', '.sample(n=10000, random_state=36)', content)
    content = re.sub(r'\.sample\(frac=[\d.]+,\s*random_state=\d+\)', '.sample(n=10000, random_state=36)', content)
    return content

def fix_output_dir(content):
    """修复输出目录"""
    # 已经由之前的脚本处理好了，这里做检查
    if '/home/cloudlab' in content and 'output_dir' in content:
        print("  Warning: 仍包含 /home/cloudlab 路径")
    return content

def fix_input_paths(content, basename):
    """修复输入数据路径"""
    maps = {
        'Texd': ('/mnt/d/AI_Seismic_Data/txed/Texd.csv', '/mnt/d/AI_Seismic_Data/txed/Texd.hdf5'),
        'Instance': ('/mnt/d/AI_Seismic_Data/Instance/INSTANCE.csv', '/mnt/d/AI_Seismic_Data/Instance/INSTANCE.hdf5'),
        'stead': ('/mnt/d/AI_Seismic_Data/STEAD/STEAD.csv', '/mnt/d/AI_Seismic_Data/STEAD/STEAD.hdf5'),
        'diting': (
            '/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.csv',
            '/mnt/d/AI_Seismic_Data/谛听1.0-50HZ/Diting50hz/processed/DiTing330km_part_26_processed.hdf5'
        ),
    }
    for key, (csv_path, h5_path) in maps.items():
        if key in basename.lower():
            # 替换所有非目标路径为正确路径
            def replace_csv(m):
                existing = m.group(3)
                if existing.startswith('/mnt/d/'):
                    return m.group(0)
                return f"{m.group(1)}{m.group(2)}{csv_path}{m.group(2)}"
            def replace_h5(m):
                existing = m.group(3)
                if existing.startswith('/mnt/d/'):
                    return m.group(0)
                return f"{m.group(1)}{m.group(2)}{h5_path}{m.group(2)}"
            content = re.sub(r'(input_csv\s*=\s*)(["\'])([^"\']+)(\2)', replace_csv, content)
            content = re.sub(r'(hdf5_path\s*=\s*)(["\'])([^"\']+)(\2)', replace_h5, content)
            break
    return content


def process_file(filename):
    abs_path = os.path.join(SCRIPT_DIR, filename)
    if not os.path.exists(abs_path):
        print(f"  SKIP: 文件不存在 - {filename}")
        return

    with open(abs_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    content = fix_hdf5_open(content)
    content = fix_init_h5file(content)
    content = fix_dataloader(content)
    content = fix_cuda(content)
    content = fix_sample(content)
    content = fix_input_paths(content, filename)
    content = fix_moe_import(content)

    # 修复缩进问题（之前修改脚本引入的）
    # 确保 if __name__ == "__main__": 下的代码缩进正确
    lines = content.split('\n')
    fixed_lines = []
    in_main_block = False
    main_indent = 0
    skip_next = 0

    for i, line in enumerate(lines):
        if skip_next > 0:
            skip_next -= 1
            continue

        # 检测 if __name__ == "__main__":
        if re.match(r'^if\s+__name__\s*==\s*["\']__main__["\']\s*:', line):
            in_main_block = True
            main_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line)
            continue

        if in_main_block:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())

            # 遇到同缩进或更小缩进的非空行（且不是else/elif），退出main块
            if stripped and indent <= main_indent and not stripped.startswith('#') and not stripped.startswith('main('):
                in_main_block = False
                fixed_lines.append(line)
                continue

            # 修复缩进：如果该行缩进不正确（小于 main_indent + 4）
            if stripped:
                if indent < main_indent + 4 and not stripped.startswith('#'):
                    # 只修复那些明显属于main块的代码行
                    if any(kw in stripped for kw in ['main(', 'start', 'end', 'time.time', 'mp.set', 'with open', 'print(', 'tf.write']):
                        line = ' ' * (main_indent + 4) + stripped

        fixed_lines.append(line)

    content = '\n'.join(fixed_lines)

    if content != original:
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  FIXED: {filename}")
    else:
        print(f"  OK:    {filename}")


def main():
    os.chdir(SCRIPT_DIR)
    for target in TARGETS:
        process_file(target)


if __name__ == "__main__":
    main()
