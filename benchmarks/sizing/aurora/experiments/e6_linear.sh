#!/bin/bash
# E6: linear-layer GEMMs (forward NT, input grad NN, weight grad TN) and the logit layer with v = 50,257 and 50,304.
export ZE_AFFINITY_MASK=0
B="python3 $QA/bench/gemm_bench.py --out $RUN/e6.csv"
for T in 4096 8192 14336 16384; do for H in 2048 3072 3584 4096 5376 7168 8192 12288 14336 16384; do
  echo "$T 4096 $H" >> $RUN/fwd.txt; echo "$T $H 4096" >> $RUN/dgrad.txt; echo "$H $T 4096" >> $RUN/wgrad.txt
done; done
for X in fwd:NT dgrad:NN wgrad:TN; do IFS=: read N L <<< "$X"
  $B --shapes $RUN/$N.txt --layout $L --tag $N > $RUN/e6_$N.log 2>&1
  ONEDNN_VERBOSE=1 python3 $QA/bench/gemm_bench.py --shapes $RUN/$N.txt --layout $L --iters 1 --warmup 0 --out $RUN/scratch.csv > $RUN/e6_verbose_$N.log 2>&1
done
for V in 50257 50304; do
  ONEDNN_VERBOSE=profile_exec $B --m 8192 --n 4096 --k $V --layout NT --iters 20 --warmup 5 --tag logits_fwd_$V > $RUN/e6_v${V}_fwd.log 2>&1
  ONEDNN_VERBOSE=profile_exec $B --m 8192 --n $V --k 4096 --layout NN --iters 20 --warmup 5 --tag logits_dgrad_$V > $RUN/e6_v${V}_dgrad.log 2>&1
  ONEDNN_VERBOSE=profile_exec $B --m $V --n 8192 --k 4096 --layout TN --iters 20 --warmup 5 --tag logits_wgrad_$V > $RUN/e6_v${V}_wgrad.log 2>&1
done
python3 $QA/analysis/compare_replay.py $RUN/e6.csv > $RUN/e6_replay_summary.txt 2>&1
