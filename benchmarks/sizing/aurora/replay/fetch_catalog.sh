#!/bin/bash
# Rebuild catalog.json from oneDNN v3.10.2.
set -e; cd "$(dirname "$0")"
curl -fsSL -o kernel.db https://raw.githubusercontent.com/uxlfoundation/oneDNN/v3.10.2/src/gpu/intel/gemm/jit/selector/db/kernel.db
python3 parse_db.py kernel.db catalog.json && rm -f kernel.db
python3 gemm_predict.py
