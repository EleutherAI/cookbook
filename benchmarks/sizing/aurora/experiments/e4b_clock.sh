#!/bin/bash
# E4b: E4 on another node, an idle-gap sweep, the original protocol (host init, short runs) and one 20,000-call run.
export ZE_AFFINITY_MASK=0
KS=""; for q in 3 5 7 9 11 13; do KS="$KS $((1792*q)) $((1792*q+32))"; done; KS="$KS 50176 50208"
B="python3 $QA/bench/gemm_bench.py --dump_iters --out $RUN/e4b.csv"
bash $QA/bench/smi_log.sh $RUN/e4b_smi.txt & sleep 15
$B --k $KS --iters 1000 --warmup 100 --tag reuse > $RUN/e4b_reuse.log 2>&1; sleep 5
$B --k $KS --iters 1000 --warmup 100 --rotate_mb 800 --tag rotate800 > $RUN/e4b_rotate.log 2>&1; sleep 5
$B --k $KS --iters 50 --warmup 10 --burst 20 --tag burst20 > $RUN/e4b_burst.log 2>&1; sleep 5
$B --m 512 1024 2048 4096 --k 16128 --iters 1000 --warmup 100 --tag vary_m > $RUN/e4b_varym.log 2>&1; sleep 5
for I in 0.2 1 3 10; do $B --k 16128 16128 16128 --idle_s $I --iters 2000 --warmup 0 --tag idle$I > $RUN/e4b_idle$I.log 2>&1; done; sleep 5
$B --k 4096 6144 8192 12288 16384 20480 24576 28672 32768 36864 40960 49152 65536 --init host --tag oldproto > $RUN/e4b_oldproto.log 2>&1; sleep 10
$B --k 16128 --iters 20000 --warmup 0 --tag long20k > $RUN/e4b_long.log 2>&1
pkill -f "xpu-smi dump"
