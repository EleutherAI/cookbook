#!/bin/bash
# E10: power limits with 1, 2, 6 and 12 busy stacks. Stack s is GPU s/2, tile s%2.
for d in 0 1 2 3 4 5; do for t in 0 1; do bash $QA/bench/smi_log.sh $RUN/smi_d${d}_t${t}.txt $d $t & done; done
sleep 20
phase() {  # name k stacks...
  local N=$1 K=$2; shift 2; local T0=$(python3 -c "import time; print(time.time() + 30)")
  echo "$(date -Is) phase $N k=$K stacks $* start_at $T0" >> $RUN/e10_phases.txt
  local P=""
  for s in "$@"; do
    ZE_AFFINITY_MASK=$s python3 $QA/bench/steady_run.py --k $K --duration 20 --start_at $T0 --out $RUN/${N}_s$s > $RUN/${N}_s$s.log 2>&1 &
    P="$P $!"
  done
  wait $P; sleep 20
}
phase p1 16128 0
phase p2 16128 0 1
phase p3 16128 0 2
phase p4 16128 0 2 4 6 8 10
phase p5 16128 $(seq 0 11)
phase p6 98562 0
phase p7 98562 $(seq 0 11)
pkill -f "xpu-smi dump"
python3 $QA/analysis/e10_analysis.py $RUN
