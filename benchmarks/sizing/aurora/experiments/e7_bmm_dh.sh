#!/bin/bash
# E7: attention-over-values BMM, out[b, s, d] = P[b, s, s] @ V[b, s, d], against d.
export ZE_AFFINITY_MASK=0
D=$(seq 32 16 512)
python3 $QA/bench/gemm_bench.py --batch 16 --m 2048 --n 2048 --k $D --out $RUN/e7.csv --tag aov_a16_s2048 > $RUN/e7.log 2>&1
python3 $QA/bench/gemm_bench.py --batch 32 --m 4096 --n 4096 --k $D --out $RUN/e7.csv --tag aov_a32_s4096 > $RUN/e7b.log 2>&1
python3 $QA/analysis/compare_replay.py $RUN/e7.csv > $RUN/e7_replay_summary.txt 2>&1
