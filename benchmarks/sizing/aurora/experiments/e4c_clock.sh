#!/bin/bash
# E4c: clock, power and temperature from idle to the sustained state (the power figure).
export ZE_AFFINITY_MASK=0
B="python3 $QA/bench/gemm_bench.py --warmup 0 --dump_iters --out $RUN/e4c.csv"
bash $QA/bench/smi_log.sh $RUN/e4c_smi_100ms.txt & sleep 15
$B --k 16128 --iters 20000 --tag long_reuse_16128 > $RUN/e4c_1.log 2>&1; sleep 15
$B --k 16128 --iters 20000 --rotate_mb 800 --tag long_rot800_16128 > $RUN/e4c_2.log 2>&1; sleep 15
$B --k 50176 --iters 8000 --tag long_reuse_50176 > $RUN/e4c_3.log 2>&1; sleep 15
python3 $QA/bench/gemm_bench.py --k 8192 49152 8192 49152 --idle_s 5 --init host --dump_iters --out $RUN/e4c.csv --tag oldproto > $RUN/e4c_4.log 2>&1; sleep 10
pkill -f "xpu-smi dump"
