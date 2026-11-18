#!/bin/bash
# 4个数据集并行运行，每个内部3个模型串行
set +e
cd /home/yuan/code/SeisMoE/benchmark

LOGDIR=logs && mkdir -p $LOGDIR

run_dataset() {
    local ds="$1" dir="$2"
    local a="${3}"; local b="${4}"; local c="${5}"
    local am="$a" bm="$b" cm="$c"
    local al="${ds}_$a" bl="${ds}_$b" cl="${ds}_$c"
    (
        for pair in "$a ${dir}_${a}" "$b ${dir}_${b}" "$c ${dir}_${c}"; do
            read -r model subdir <<< "$pair"
            mkdir -p "$subdir"
            script="seis_${dir}_${model}.py"
            [ "$dir" = "instance" ] && script="seis_instance_${model}.py"
            echo "[$ds/$model] start @ $(date +%H:%M:%S)"
            python "$script" > "$LOGDIR/${ds}_${model}.log" 2>&1
            RC=$?
            TF="$subdir/time.txt"
            if [ -f "$TF" ]; then
                echo "[$ds/$model] DONE  $(cat "$TF")s" 
                echo "${ds}_${model}: $(cat "$TF")" >> "$LOGDIR/results.txt"
            else
                echo "[$ds/$model] FAIL (rc=$RC)"
                echo "${ds}_${model}: FAIL" >> "$LOGDIR/results.txt"
            fi
        done
    ) &
}

# Txed (1万) + DiTing (1万) 快 -> 先并行跑
run_dataset "Txed"     "Texd"     "Eqtransformer" "phasenet" "Moe"
run_dataset "DiTing"   "diting"   "Eqtransformer" "PhaseNet" "Moe"

# Instance (10万) + STEAD (10万) 慢 -> 并行跑
run_dataset "Instance" "Instance" "Eqtransformer" "PhaseNet" "Moe"
run_dataset "STEAD"    "stead"    "Eqtransformer" "PhaseNet" "Moe"

wait

echo "=== ALL DONE ==="
cat "$LOGDIR/results.txt"

python3 -c "
import os
dirs = [
    ('Texd','Txed','Eqtransformer'),('Texd','Txed','phasenet'),('Texd','Txed','Moe'),
    ('Instance','Instance','Eqtransformer'),('Instance','Instance','PhaseNet'),('Instance','Instance','Moe'),
    ('diting','DiTing','Eqtransformer'),('diting','DiTing','PhaseNet'),('diting','DiTing','Moe'),
    ('stead','STEAD','Eqtransformer'),('stead','STEAD','PhaseNet'),('stead','STEAD','Moe'),
]
r={}
for sub,ds,m in dirs:
    tf=os.path.join(f'{sub}_{m}','time.txt')
    r[f'{ds}_{m}'] = float(open(tf).read().strip()) if os.path.exists(tf) else None

# 处理 instance_Moe 特殊情况
tf2='Instance_Moe/time.txt'
if os.path.exists(tf2):
    r['Instance_MoE']=float(open(tf2).read().strip())

md='''# 地震波形拾取模型运行时间对比\n
| 数据集 | 数据量 | EQTransformer | PhaseNet | MoE |
|--------|--------|:-----------:|:--------:|:---:|\n'''
counts={'Txed':'1万','DiTing':'1万','Instance':'10万','STEAD':'10万'}
for ds in ['Txed','Instance','DiTing','STEAD']:
    vals=[]
    for m in ['EQTransformer','PhaseNet','MoE']:
        v=r.get(f'{ds}_{m}')
        vals.append(f'{v:.1f}s' if v else 'N/A')
    md+=f\"| {ds} | {counts[ds]} | {vals[0]} | {vals[1]} | {vals[2]} |\\n\"
md+='\n> GPU: 8GB | 内存: 64GB | CPU: 24核\n'
with open('README.md','w') as f: f.write(md)
print(md)
"