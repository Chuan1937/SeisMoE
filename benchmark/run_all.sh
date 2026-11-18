#!/bin/bash
set +e
RESULT_FILE="all_times.json"
echo "{" > $RESULT_FILE

run_one() {
    local label="$1" script="$2"
    echo "=== $label ==="
    START=$(date +%s)
    python "$script" 2>&1 | tail -3
    EXIT=$?
    END=$(date +%s)
    if [ $EXIT -eq 0 ]; then
        TIME_FILE="$(dirname "$script" 2>/dev/null || echo '.')/${label%%/*}/time.txt"
        # try the actual output dir
        SUBDIR=$(echo "$label" | sed 's/.*\///')
        for d in Texd_Eqtransformer Texd_phasenet Texd_Moe \
                 Instance_Eqtransformer Instance_PhaseNet Instance_Moe \
                 diting_Eqtransformer diting_PhaseNet diting_Moe \
                 stead_Eqtransformer stead_PhaseNet stead_Moe; do
            [ -f "$d/time.txt" ] && ELAPSED=$(cat "$d/time.txt") && break
        done
        ELAPSED=${ELAPSED:-$((END-START))}
        echo "  TIME: ${ELAPSED}s"
        echo "  \"${label}\": ${ELAPSED}," >> $RESULT_FILE
    else
        echo "  FAILED"
        echo "  \"${label}\": null," >> $RESULT_FILE
    fi
    echo ""
}

# Txed - 1万条 (约10秒每个)
run_one "Txed/EQTrans"    seis_Texd_Eqtransformer.py
run_one "Txed/PhaseNet"   seis_Texd_phasenet.py
run_one "Txed/MoE"        seis_Texd_Moe.py

# Instance - 10万条 (较慢)
run_one "Instance/EQTrans"  seis_Instance_Eqtransformer.py
run_one "Instance/PhaseNet" seis_Instance_PhaseNet.py
run_one "Instance/MoE"      seis_instance_Moe.py

# DiTing - 1万条
run_one "DiTing/EQTrans"   seis_diting_Eqtransformer.py
run_one "DiTing/PhaseNet"  seis_diting_PhaseNet.py
run_one "DiTing/MoE"       seis_diting_Moe.py

# STEAD - 10万条 (慢)
run_one "STEAD/EQTrans"   seis_stead_Eqtransformer.py
run_one "STEAD/PhaseNet"  seis_stead_PhaseNet.py
run_one "STEAD/MoE"       seis_stead_Moe.py

# Fix trailing comma and close JSON
sed -i '$ s/,$//' $RESULT_FILE
echo "}" >> $RESULT_FILE
echo "=== ALL DONE ==="
cat $RESULT_FILE
