#!/bin/bash
# E9 chunk @N@, written by analysis/e9_plan.py: steady-state K sweep (m = n = 2048), then the M=K sweep if time is left.
export ZE_AFFINITY_MASK=0
bash $QA/bench/smi_log.sh $RUN/e9_smi_500ms.txt 0 0 500 &
[ -s @KFILE@ ] && python3 $QA/bench/steady_sweep.py --points @KFILE@ --out $RUN/e9k --tag e9_c@N@ --budget_s 3050 > $RUN/e9k.log 2>&1
DONE=$(python3 -c "import json; print(int(json.load(open('$RUN/e9k_status.json'))['stop'] == 'complete'))" 2>/dev/null || echo 1)
if [ "$DONE" = 1 ] && [ -s @MFILE@ ] && [ $SECONDS -lt 2550 ]; then
  python3 $QA/bench/steady_sweep.py --points @MFILE@ --out $RUN/e9mk --tag e9_mk_c@N@ --budget_s 600 > $RUN/e9mk.log 2>&1
fi
pkill -f "xpu-smi dump"
