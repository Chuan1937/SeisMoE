#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量修复所有优化脚本的缩进问题
"""

import os
import glob
import re

def fix_script(filepath):
    """修复脚本中的缩进问题"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复output = model(data)的缩进
    # 查找模式：有多个空行后跟缩进的output = model(data)
    pattern = r'(\s+data = batch\[\'data\'\]\.to\(device, non_blocking=True\))\s+\n\s+\n(\s+output = model\(data\))'
    replacement = r'\1\n            \n\2'
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"修复: {filepath}")
        return True
    else:
        print(f"无需修复: {filepath}")
        return False

def main():
    scripts = glob.glob("*_optimized.py")
    fixed_count = 0
    
    for script in scripts:
        if fix_script(script):
            fixed_count += 1
    
    print(f"\n修复了 {fixed_count} 个脚本")

if __name__ == "__main__":
    main()