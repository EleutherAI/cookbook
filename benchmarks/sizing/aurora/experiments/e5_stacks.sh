#!/bin/bash
# E5: the same period on each of the 12 stacks of one node.
for S in $(seq 0 11); do
  ZE_AFFINITY_MASK=$S python3 $QA/bench/gemm_bench.py --k_range 41216 43010 2 --iters 50 --warmup 10 --out $RUN/e5.csv --tag stack$S > $RUN/e5_stack$S.log 2>&1
done
