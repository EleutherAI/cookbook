#!/bin/bash
# E6b: E6 in steady state; each grid starts with a pre-heat shape and runs forward, then in reverse order.
export ZE_AFFINITY_MASK=0
B="python3 $QA/bench/gemm_bench.py --out $RUN/e6b.csv"
H0="16384 4096 16384"
for T in 4096 8192 14336 16384; do for H in 2048 3072 3584 4096 5376 7168 8192 12288 14336 16384; do
  echo "$T 4096 $H" >> $RUN/fwd.txt; echo "$T $H 4096" >> $RUN/dgrad.txt; echo "$H $T 4096" >> $RUN/wgrad.txt
done; done
for X in fwd:NT dgrad:NN wgrad:TN; do IFS=: read N L <<< "$X"
  { echo "$H0"; cat $RUN/$N.txt; echo "$H0"; tac $RUN/$N.txt; } > $RUN/e6b_${N}_shapes.txt
  $B --shapes $RUN/e6b_${N}_shapes.txt --layout $L --tag $N > $RUN/e6b_$N.log 2>&1
done
for V in 50257 50304; do
  $B --m 16384 8192 8192 8192 --n 4096 --k $V --layout NT --iters 100 --warmup 20 --tag logits_fwd_$V > $RUN/e6b_v${V}_fwd.log 2>&1
  $B --shapes <(echo "$H0"; echo "8192 $V 4096"; echo "8192 $V 4096") --layout NN --iters 100 --warmup 20 --tag logits_dgrad_$V > $RUN/e6b_v${V}_dgrad.log 2>&1
  $B --shapes <(echo "$H0"; echo "$V 8192 4096"; echo "$V 8192 4096") --layout TN --iters 100 --warmup 20 --tag logits_wgrad_$V > $RUN/e6b_v${V}_wgrad.log 2>&1
done
python3 $QA/analysis/compare_replay.py $RUN/e6b.csv > $RUN/e6b_replay_summary.txt 2>&1
