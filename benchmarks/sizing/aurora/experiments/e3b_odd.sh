#!/bin/bash
# E3b: odd k over the same period.
export ZE_AFFINITY_MASK=0
python3 $QA/bench/gemm_bench.py --k_range 98305 100099 14 --out $RUN/e3b_odd.csv --tag e3b_odd > $RUN/e3b_odd.log 2>&1
python3 $QA/bench/gemm_bench.py --k_range 98305 100099 98 --init host --out $RUN/e3b_odd_oldstyle.csv --tag e3b_odd_old > $RUN/e3b_odd_old.log 2>&1
