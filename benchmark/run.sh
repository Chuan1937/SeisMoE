#!/bin/bash
echo "=== 地震模型时间对比 ==="
mkdir -p Texd_Eqtransformer Texd_phasenet Texd_Moe
mkdir -p Instance_Eqtransformer Instance_PhaseNet Instance_Moe
mkdir -p diting_Eqtransformer diting_PhaseNet diting_Moe
mkdir -p stead_Eqtransformer stead_PhaseNet stead_Moe

for s in seis_Texd_Eqtransformer.py seis_Texd_phasenet.py seis_Texd_Moe.py \
         seis_Instance_Eqtransformer.py seis_Instance_PhaseNet.py seis_instance_Moe.py \
         seis_diting_Eqtransformer.py seis_diting_PhaseNet.py seis_diting_Moe.py \
         seis_stead_Eqtransformer.py seis_stead_PhaseNet.py seis_stead_Moe.py; do
    echo "=== $s ==="
    python "$s" || echo "FAILED: $s"
    echo ""
done

# 收集时间并生成markdown
python3 -c "
import os, json
dirs=['Texd_Eqtransformer','Texd_phasenet','Texd_Moe',
      'Instance_Eqtransformer','Instance_PhaseNet','Instance_Moe',
      'diting_Eqtransformer','diting_PhaseNet','diting_Moe',
      'stead_Eqtransformer','stead_PhaseNet','stead_Moe']
r={}
for d in dirs:
    tf=os.path.join(d,'time.txt')
    if os.path.exists(tf):
        with open(tf) as f: r[d]=float(f.read().strip())
    else: r[d]='N/A'
print(json.dumps(r,indent=2))
"
echo "=== DONE ==="
