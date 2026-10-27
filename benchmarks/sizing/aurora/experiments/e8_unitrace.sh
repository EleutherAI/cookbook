#!/bin/bash
# E8: unitrace hardware counters for the same kernel at 64, 32, 16 and 8 byte row alignment.
export ZE_AFFINITY_MASK=0
python3 $QA/bench/gemm_bench.py --k 98560 98560 98564 98568 98576 98560 --iters 400 --warmup 100 --out $RUN/e8_timing.csv > $RUN/e8_timing.log 2>&1
unitrace --metric-list > $RUN/e8_metric_list.txt 2>&1
for G in ComputeBasic L1ProfileReads L1ProfileWrites L1ProfileSlmBankConflicts L1Cache1 L1Cache2 L1Cache3 L1Cache4 L1Cache5 L1Cache6; do
  for K in 98560 98564 98568 98576; do
    timeout 120 unitrace -q -g $G python3 $QA/bench/gemm_bench.py --k $K --iters 20 --warmup 10 --out $RUN/e8_bench.csv --tag ${G}_k$K > $RUN/ut_${G}_k$K.log 2>&1
  done
done
python3 $QA/analysis/e8_parse.py $RUN
