#!/bin/bash
# Usage: bench/env_capture.sh <outdir>
OUT=${1:?outdir}; mkdir -p "$OUT"
{
hostname; date -Is; module list 2>&1
env | grep -E '^(PBS_|ZE_|ONEDNN|SYCL|ONEAPI|UR_)' | sort
python3 -c "import torch; print(torch.__version__, torch.version.xpu); [print(i, torch.xpu.get_device_properties(i)) for i in range(torch.xpu.device_count())]"
python3 -m pip list 2>/dev/null | grep -i -E 'torch|intel|triton|oneccl'
xpu-smi discovery 2>&1 | head -60
ONEDNN_VERBOSE=1 python3 -c "import torch; a = torch.randn(256, 256, device='xpu', dtype=torch.bfloat16); a @ a; torch.xpu.synchronize()" 2>&1 | grep info
} > "$OUT/env_$(hostname).txt" 2>&1
