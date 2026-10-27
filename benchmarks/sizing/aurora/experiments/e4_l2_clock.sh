#!/bin/bash
# E4: L2 residency (--rotate_mb) against run length (--burst), with clock and power.
export ZE_AFFINITY_MASK=0
KS=""; for q in 3 5 7 9 11 13; do KS="$KS $((1792*q)) $((1792*q+32))"; done; KS="$KS 50176 50208"
B="python3 $QA/bench/gemm_bench.py --iters 1000 --warmup 100 --dump_iters --out $RUN/e4.csv"
bash $QA/bench/smi_log.sh $RUN/e4_smi.txt & sleep 5
$B --k $KS --tag reuse > $RUN/e4_reuse.log 2>&1; sleep 5
$B --k $KS --rotate_mb 800 --tag rotate800 > $RUN/e4_rotate.log 2>&1; sleep 5
python3 $QA/bench/gemm_bench.py --k $KS --burst 20 --iters 50 --warmup 10 --dump_iters --out $RUN/e4.csv --tag burst20 > $RUN/e4_burst.log 2>&1; sleep 5
$B --m 512 1024 2048 4096 --k 16128 --tag vary_m > $RUN/e4_varym.log 2>&1
pkill -f "xpu-smi dump"
