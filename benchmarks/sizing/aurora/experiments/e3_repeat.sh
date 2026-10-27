#!/bin/bash
# E3: one 1792-wide period at even k, compared with the selector replay; a subset with host-side operand creation.
export ZE_AFFINITY_MASK=0
python3 $QA/bench/gemm_bench.py --k_range 98304 100098 2 --out $RUN/e3.csv --tag e3_new > $RUN/e3.log 2>&1
python3 $QA/bench/gemm_bench.py --k_range 98304 100098 14 --init host --out $RUN/e3_oldstyle.csv --tag e3_old > $RUN/e3_old.log 2>&1
python3 $QA/analysis/compare_replay.py $RUN/e3.csv > $RUN/e3_replay_summary.txt 2>&1
