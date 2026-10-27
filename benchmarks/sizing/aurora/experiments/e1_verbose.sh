#!/bin/bash
# E1: oneDNN implementation per row alignment of k.
export ZE_AFFINITY_MASK=0
for K in 98560 98561 98562 98564 99330 41218 41986; do
  ONEDNN_VERBOSE=profile_exec python3 $QA/bench/gemm_bench.py --k $K --iters 5 --warmup 2 --out $RUN/e1.csv > $RUN/e1_k$K.log 2>&1
done
ONEDNN_VERBOSE=all python3 $QA/bench/gemm_bench.py --k 98562 --iters 1 --warmup 0 --out $RUN/e1_all.csv > $RUN/e1_all.log 2>&1
{ grep -h "info,oneDNN" $RUN/e1_k*.log | sort -u; grep -h ",exec,gpu,matmul" $RUN/e1_k*.log; } > $RUN/e1_summary.txt
