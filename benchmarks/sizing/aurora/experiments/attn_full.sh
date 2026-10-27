#!/bin/bash
# Attention, full run: attn_bench.py subcommands in steady state, then kernel identification for SDPA over d and for Triton tiles.
export ZE_AFFINITY_MASK=0; source $QA/experiments/attn_env.sh
bash $QA/bench/smi_log.sh $RUN/smi_100ms.txt &
for SUB in micro tricfg sdpa ringx; do
  timeout 1200 python3 $QA/attn/attn_bench.py $SUB --out $RUN/attn > $RUN/attn_$SUB.log 2>&1
done
for D in 16 32 48 64 80 96 128 160 192 256 384 512; do
  ONEDNN_VERBOSE=1 timeout 120 unitrace -d -v python3 $QA/attn/attn_id.py --sdpa_d $D --tri > $RUN/id_sdpa_d$D.log 2>&1
done
for T in 32:128:32:8:default 64:128:64:16:default 128:128:32:8:default 128:128:64:16:default 256:64:64:8:default 256:128:32:16:default; do
  timeout 120 unitrace -d -v python3 $QA/attn/attn_id.py --sdpa_d --tri $T > $RUN/id_tri_${T//:/_}.log 2>&1
done
pkill -f "xpu-smi dump"
python3 $QA/attn/analyze_attn.py $RUN
