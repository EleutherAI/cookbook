#!/bin/bash
# Attention smoke test: every attn_bench.py subcommand on tiny grids, plus kernel identification.
export ZE_AFFINITY_MASK=0; source $QA/experiments/attn_env.sh
bash $QA/bench/smi_log.sh $RUN/smi_100ms.txt &
for SUB in micro tricfg sdpa ringx; do
  timeout 400 python3 $QA/attn/attn_bench.py $SUB --smoke --settle_s 0.2 --meas_s 0.2 --out $RUN/smoke > $RUN/smoke_$SUB.log 2>&1
done
ONEDNN_VERBOSE=1 timeout 300 unitrace -d -v python3 $QA/attn/attn_id.py > $RUN/attn_id.log 2>&1
pkill -f "xpu-smi dump"
