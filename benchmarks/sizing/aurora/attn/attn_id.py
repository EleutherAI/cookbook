#!/usr/bin/env python3
"""A few calls of SDPA and of the Triton FA-2 forward, to run under ONEDNN_VERBOSE or unitrace -d -v."""
import argparse, os, sys
import torch, torch.nn.functional as F, triton
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tri_fa_kernels import _attn_fwd

p = argparse.ArgumentParser()
p.add_argument("--tri", nargs="*", default=["64:128:64:8:default", "128:128:64:8:default", "256:64:64:8:default"])
p.add_argument("--sdpa_d", type=int, nargs="*", default=[64, 80, 128, 256])
a = p.parse_args()
dev = torch.device("xpu", 0)
triton.set_allocator(lambda size, align, _: torch.empty(size, dtype=torch.int8, device=dev))
B, A, S = 4, 32, 2048
for d, causal in [(d, False) for d in a.sdpa_d] + ([(128, True)] if 128 in a.sdpa_d else []):
    q, k, v = [torch.randn(B, A, S, d, device=dev, dtype=torch.bfloat16) for _ in range(3)]
    print(f"### SDPA d={d} causal={causal}", flush=True)
    for _ in range(3):
        F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.xpu.synchronize()
for spec in a.tri:
    d, bm, bn, w, g = spec.split(":"); d, bm, bn, w = int(d), int(bm), int(bn), int(w)
    q, k, v = [torch.randn(B, A, S, d, device=dev, dtype=torch.bfloat16) for _ in range(3)]
    o = torch.empty_like(q); M = torch.empty((B, A, S), device=dev, dtype=torch.float32)
    print(f"### TRITON d={d} BM={bm} BN={bn} warps={w} grf={g}", flush=True)
    for _ in range(3):
        kern = _attn_fwd[(triton.cdiv(S, bm), B * A, 1)](d ** -0.5, M, B, A, q, k, v, o, N_CTX_Q=S, N_CTX_K=S, HEAD_DIM=d,
                                                         BLOCK_M=bm, BLOCK_N=bn, FP8_OUTPUT=False, USE_BF16=True, STAGE=1,
                                                         warp_specialize=True, IS_HOPPER=False, num_warps=w, num_stages=2, grf_mode=g)
    torch.xpu.synchronize()
    md = kern.metadata
    print(f"    n_regs={kern.n_regs} n_spills={kern.n_spills} threads_per_warp={getattr(md, 'threads_per_warp', None)} "
          f"shared={getattr(md, 'shared', None)} build_flags={getattr(md, 'build_flags', '')!r}", flush=True)
