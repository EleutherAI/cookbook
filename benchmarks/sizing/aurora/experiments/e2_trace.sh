#!/bin/bash
# E2: kernel names, launch grid and extra operations per call.
export ZE_AFFINITY_MASK=0
for K in 98560 98561 98562 99330; do
  python3 $QA/bench/gemm_bench.py --k $K --iters 5 --warmup 3 --profile --out $RUN/e2.csv > $RUN/e2_profiler_k$K.log 2>&1
  unitrace -d -v python3 $QA/bench/gemm_bench.py --k $K --iters 3 --warmup 1 --out $RUN/e2u.csv > $RUN/e2_unitrace_k$K.log 2>&1
done
