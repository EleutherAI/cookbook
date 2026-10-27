#!/bin/bash
# Usage: bench/smi_log.sh <out> [gpu] [tile] [ms] &   stop with: pkill -f "xpu-smi dump"
# --ims needs --file, and --file needs a terminal, so xpu-smi runs under script(1) with an open stdin.
OUT=${1:?out}; D=${2:-0}; T=${3:-0}; MS=${4:-100}
sleep 86400 | script -qfec "xpu-smi dump -d $D -t $T -m 0,1,2,3,35 --ims $MS --file $OUT --date" /dev/null > $OUT.stdout 2>&1
