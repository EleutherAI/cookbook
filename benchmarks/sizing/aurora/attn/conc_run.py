#!/usr/bin/env python3
"""SDPA and Triton FA-2 for a fixed time from a common start time, for concurrent runs on several stacks."""
import argparse, json, os, socket, sys, time
import numpy as np, torch, torch.nn.functional as F, triton
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tri_fa_kernels import _attn_fwd

p = argparse.ArgumentParser(); p.add_argument("--start_at", type=float, required=True); p.add_argument("--out", required=True)
p.add_argument("--seconds", type=float, default=3.0)
a = p.parse_args()
dev = torch.device("xpu", 0)
triton.set_allocator(lambda size, align, _: torch.empty(size, dtype=torch.int8, device=dev))
B, A, S = 4, 32, 2048
plan = []
for d in (64, 128, 256):
    q, k, v = [torch.randn(B, A, S, d, device=dev, dtype=torch.bfloat16) for _ in range(3)]
    plan.append((f"sdpa_d{d}", d, (lambda q=q, k=k, v=v: F.scaled_dot_product_attention(q, k, v))))
    o = torch.empty_like(q); M = torch.empty((B, A, S), device=dev, dtype=torch.float32)
    w = 16 if d == 256 else 8
    plan.append((f"triton_d{d}", d, (lambda q=q, k=k, v=v, o=o, M=M, d=d, w=w: _attn_fwd[(S // 128, B * A, 1)](
        d ** -0.5, M, B, A, q, k, v, o, N_CTX_Q=S, N_CTX_K=S, HEAD_DIM=d, BLOCK_M=128, BLOCK_N=32, FP8_OUTPUT=False,
        USE_BF16=True, STAGE=1, warp_specialize=True, IS_HOPPER=False, num_warps=w, num_stages=2))))
for _, _, fn in plan:  # compile
    for _ in range(3):
        fn()
torch.xpu.synchronize()
late = time.time() - a.start_at
while time.time() < a.start_at:
    time.sleep(0.001)
e0, e1 = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
res = []
for name, d, fn in plan:
    t0 = time.time(); ts = []
    while time.time() - t0 < a.seconds:
        e0.record(); fn(); e1.record(); torch.xpu.synchronize()
        ts.append((time.time() - t0, e0.elapsed_time(e1)))
    ms = np.array([m for t, m in ts if t >= 0.5])
    flop = 4.0 * B * A * S * S * d
    res.append(dict(kernel=name, d=d, calls=len(ms), tflops=flop / (np.median(ms) * 1e-3) / 1e12, host_t0=t0, host_t1=time.time(),
                    mask=os.environ.get("ZE_AFFINITY_MASK"), host=socket.gethostname(), ready_late_s=round(late, 2)))
with open(a.out, "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in res)
print("\n".join(json.dumps(r) for r in res), flush=True)
