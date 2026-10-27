#!/bin/bash
# Attention on stack 0 alone, then on all 12 stacks at once.
source $QA/experiments/attn_env.sh
for d in 0 1 2 3 4 5; do for t in 0 1; do bash $QA/bench/smi_log.sh $RUN/smi_d${d}_t${t}.txt $d $t & done; done
ZE_AFFINITY_MASK=0 python3 $QA/attn/conc_run.py --start_at 0 --seconds 0.3 --out $RUN/precompile.jsonl > $RUN/precompile.log 2>&1
sleep 20
ZE_AFFINITY_MASK=0 python3 $QA/attn/conc_run.py --start_at $(python3 -c "import time; print(time.time() + 60)") --out $RUN/alone_s0.jsonl > $RUN/alone_s0.log 2>&1
sleep 20
T0=$(python3 -c "import time; print(time.time() + 90)")
P=""
for s in $(seq 0 11); do
  ZE_AFFINITY_MASK=$s python3 $QA/attn/conc_run.py --start_at $T0 --out $RUN/all12_s$s.jsonl > $RUN/all12_s$s.log 2>&1 &
  P="$P $!"
done
wait $P
pkill -f "xpu-smi dump"
