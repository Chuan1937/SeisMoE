#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全面修复所有优化脚本
"""

import os
import glob

def fix_all_scripts():
    """修复所有优化脚本"""
    scripts = glob.glob("*_optimized.py")
    
    for script_path in scripts:
        print(f"处理: {script_path}")
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 修复1: 移除autocast相关代码
        if "torch.cuda.amp.autocast" in content:
            # 删除with语句行
            lines = content.split('\n')
            new_lines = []
            for line in lines:
                if "torch.cuda.amp.autocast" in line:
                    continue
                new_lines.append(line)
            content = '\n'.join(new_lines)
        
        # 修复2: 修复output = model(data)的缩进
        # 查找模式：data = batch['data'].to(...) 后面有多个空行，然后是缩进的output = model(data)
        import re
        pattern = r'(data = batch\[\'data\'\]\.to\(device, non_blocking=True\))\s+\n\s+\n(\s+output = model\(data\))'
        replacement = r'\1\n            \n\2'
        content = re.sub(pattern, replacement, content)
        
        # 修复3: 确保model.load_state_dict的缩进正确
        pattern = r'(model = IMG1\(\)\.to\(device\))\n([^\s].*model\.load_state_dict)'
        replacement = r'\1\n    \2'
        content = re.sub(pattern, replacement, content)
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  修复完成")

if __name__ == "__main__":
    fix_all_scripts()