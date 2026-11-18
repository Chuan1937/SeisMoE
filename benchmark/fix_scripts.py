#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复优化脚本中的缩进问题
"""

import os
import glob

def fix_indentation(filepath):
    """修复文件中的缩进"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复autocast相关的缩进
    old_pattern = """            with torch.cuda.amp.autocast(enabled=True):
                output = model(data)"""
    
    new_pattern = """            output = model(data)"""
    
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"修复: {filepath}")
    else:
        print(f"无需修复: {filepath}")

def main():
    scripts = glob.glob("*_optimized.py")
    for script in scripts:
        fix_indentation(script)

if __name__ == "__main__":
    main()