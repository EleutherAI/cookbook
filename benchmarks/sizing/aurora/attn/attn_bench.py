#!/usr/bin/env python3
"""Attention on one stack. sdpa: F.scaled_dot_product_attention; ringx: ringX Triton kernel with its autotuner;
tricfg: the same kernel at fixed tiles; micro: FMA, exp2, DPAS and the FA-2 inner loop. Usage: attn_bench.py <sub> --out <prefix>"""
import argparse, json, os, socket, sys, time, traceback
import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tri_fa_kernels import _attn_fwd

p = argparse.ArgumentParser()
p.add_argument("sub", choices=["sdpa", "ringx", "tricfg", "micro"])
p.add_argument("--out", required=True)
p.add_argument("--smoke", action="store_true", help="tiny grids, to check that everything compiles and runs")
p.add_argument("--settle_s", type=float, default=0.5)
p.add_argument("--meas_s", type=float, default=0.5)
args = p.parse_args()
dev = torch.device("xpu", 0)
bf16 = torch.bfloat16
triton.set_allocator(lambda size, align, _: torch.empty(size, dtype=torch.int8, device=dev))
host = socket.gethostname()
OUT = f"{args.out}_{args.sub}.jsonl"
rows_written = [0]


def write(row):
    row = dict(row, sub=args.sub, host=host)
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    rows_written[0] += 1
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)


ev0, ev1 = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)


def measure(fn, flop, min_calls=10):
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    t0 = time.time()
    while time.time() - t0 < args.settle_s:
        fn(); torch.xpu.synchronize()
    ts, h0 = [], time.time()
    while (time.time() - h0 < args.meas_s or len(ts) < min_calls) and not (len(ts) >= 3 and time.time() - h0 > 5):
        ev0.record(); fn(); ev1.record(); torch.xpu.synchronize()
        ts.append(ev0.elapsed_time(ev1))
    ts = np.array(ts)
    tf = lambda ms: flop / (ms * 1e-3) / 1e12
    return dict(calls=len(ts), median_ms=float(np.median(ts)), tflops=tf(np.median(ts)), tflops_p10t=tf(np.percentile(ts, 10)),
                tflops_p90t=tf(np.percentile(ts, 90)), host_t0=h0, host_t1=time.time())


def gemm_ceiling(tag, seconds=2.0):
    A = torch.randn(2048, 2048, device=dev, dtype=bf16); B = torch.randn(2048, 16128, device=dev, dtype=bf16)
    C = torch.empty(2048, 16128, device=dev, dtype=bf16)
    t0, n = time.time(), 0
    while time.time() - t0 < seconds:
        for _ in range(10):
            torch.mm(A, B, out=C)
        torch.xpu.synchronize(); n += 10
    r = measure(lambda: torch.mm(A, B, out=C), 2.0 * 2048 * 2048 * 16128)
    write(dict(kind="gemm_ceiling", tag=tag, **r))


def attn_flop(b, a, s, d, causal):
    return 4.0 * b * a * s * s * d * (0.5 if causal else 1.0)


def qkv(b, a, s, d):
    return [torch.randn(b, a, s, d, device=dev, dtype=bf16) for _ in range(3)]


def run_sdpa():
    B = 4
    shapes = []  # (a, s, d, causal, group)
    ds = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256, 384, 512] if not args.smoke else [64, 80, 128]
    shapes += [(32, 2048, d, False, "d_sweep") for d in ds]
    shapes += [(32, 2048, d, True, "d_sweep_causal") for d in ([64, 128, 256] if not args.smoke else [64])]
    if not args.smoke:
        shapes += [(a, 2048, 64, False, "a_sweep_d64") for a in (8, 16, 64, 128)]
        shapes += [(a, 2048, 128, False, "a_sweep_d128") for a in (8, 16, 64, 128)]
        shapes += [(32, s, d, c, "s_sweep") for d in (64, 128) for s in (4096, 8192, 16384) for c in (False, True)]
    for a, s, d, causal, grp in shapes:
        try:
            q, k, v = qkv(B, a, s, d)
            r = measure(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal), attn_flop(B, a, s, d, causal))
            write(dict(kind="sdpa", tag=grp, b=B, a=a, s=s, d=d, causal=int(causal), **r))
            del q, k, v
        except Exception as e:
            write(dict(kind="sdpa", tag=grp, b=B, a=a, s=s, d=d, causal=int(causal), error=repr(e)[:200]))
        torch.xpu.empty_cache()


def run_ringx():
    os.environ.pop("TRITON_AUTOTUNE_CACHE_DIR", None)
    import ringX_attn.fused_attention as rfa
    api = rfa.get_backend_api()
    B = 4
    shapes = [(32, 2048, d, False) for d in ((64, 128) if args.smoke else (64, 128, 256))]
    if not args.smoke:
        shapes += [(32, 2048, d, True) for d in (64, 128)] + [(32, 8192, 128, False)]
    for a, s, d, causal in shapes:
        try:
            q, k, v = qkv(B, a, s, d)
            t = time.time(); api.forward(q, k, v, causal, d ** -0.5); torch.xpu.synchronize(); tune_s = time.time() - t
            best = getattr(rfa._attn_fwd, "best_config", None)
            r = measure(lambda: api.forward(q, k, v, causal, d ** -0.5), attn_flop(B, a, s, d, causal))
            write(dict(kind="ringx_headfirst", b=B, a=a, s=s, d=d, causal=int(causal), autotune_s=round(tune_s, 1),
                       best_config=str(best), **r))
            qs, ks, vs = [x.transpose(1, 2).contiguous() for x in (q, k, v)]  # (b, s, a, d)

            def with_permutes():  # as in ringX local_attn_forward
                o, lse = api.forward(qs.permute(0, 2, 1, 3).contiguous(), ks.permute(0, 2, 1, 3).contiguous(),
                                     vs.permute(0, 2, 1, 3).contiguous(), causal, d ** -0.5)
                return o.permute(0, 2, 1, 3).contiguous()
            r = measure(with_permutes, attn_flop(B, a, s, d, causal))
            write(dict(kind="ringx_with_permutes", b=B, a=a, s=s, d=d, causal=int(causal), best_config=str(best), **r))
            del q, k, v, qs, ks, vs
        except Exception as e:
            write(dict(kind="ringx", b=B, a=a, s=s, d=d, causal=int(causal), error=repr(e)[:300]))
        torch.xpu.empty_cache()


def tri_fa(q, k, v, causal, BM, BN, warps, stages, grf):
    Z, H, N, D = q.shape
    o = torch.empty_like(q)
    M = torch.empty((Z, H, N), device=dev, dtype=torch.float32)
    kern = _attn_fwd[(triton.cdiv(N, BM), Z * H, 1)](
        D ** -0.5, M, Z, H, q, k, v, o, N_CTX_Q=N, N_CTX_K=N, HEAD_DIM=D, BLOCK_M=BM, BLOCK_N=BN, FP8_OUTPUT=False,
        USE_BF16=True, STAGE=3 if causal else 1, warp_specialize=True, IS_HOPPER=False,
        num_warps=warps, num_stages=stages, grf_mode=grf)
    return o, kern


def kern_info(kern):
    md = getattr(kern, "metadata", None)
    bf = getattr(md, "build_flags", "") if md is not None else ""
    return dict(n_regs=getattr(kern, "n_regs", None), n_spills=getattr(kern, "n_spills", None),
                grf_256=int("256-GRF" in bf), build_flags=bf.strip()[:120], shared=getattr(md, "shared", None),
                threads_per_warp=getattr(md, "threads_per_warp", None))


def run_tricfg():
    B, A, S = 4, 32, 2048
    if args.smoke:
        grid = [(d, bm, bn, w, 2, g) for d in (64, 128) for bm in (128,) for bn in (64,) for w in (8, 16) for g in ("default", "256")]
    else:
        grid = [(d, bm, bn, w, 2, g) for d in (32, 64, 128, 256) for bm in (64, 128, 256) for bn in (32, 64, 128) if bn <= d
                for w in (8, 16, 32) for g in ("default",)]
    q = k = v = None; cur = None
    best = {}
    for d, bm, bn, w, st, g in grid:
        if cur != d:
            q, k, v = qkv(B, A, S, d); cur = d
        row = dict(kind="tri_fa", b=B, a=A, s=S, d=d, causal=0, BLOCK_M=bm, BLOCK_N=bn, num_warps=w, num_stages=st, grf_mode=g)
        try:
            _, kern = tri_fa(q, k, v, False, bm, bn, w, st, g)
            torch.xpu.synchronize()
            r = measure(lambda: tri_fa(q, k, v, False, bm, bn, w, st, g), attn_flop(B, A, S, d, False))
            write(dict(row, **kern_info(kern), **r))
            if r["tflops"] > best.get(d, (0, None))[0]:
                best[d] = (r["tflops"], (bm, bn, w, g))
        except Exception as e:
            write(dict(row, error=repr(e)[:300]))
    # best tile per d: more stages, small GRF, causal, longer s
    for d, (tfv, (bm, bn, w, g)) in sorted(best.items()):
        for st, gg, causal, ss in [(3, g, False, S), (4, g, False, S), (2, "128", False, S), (2, g, True, S), (2, g, False, 8192),
                                   (2, g, True, 8192)]:
            q, k, v = qkv(B, A, ss, d)
            row = dict(kind="tri_fa_followup", b=B, a=A, s=ss, d=d, causal=int(causal), BLOCK_M=bm, BLOCK_N=bn, num_warps=w,
                       num_stages=st, grf_mode=gg)
            try:
                _, kern = tri_fa(q, k, v, causal, bm, bn, w, st, gg); torch.xpu.synchronize()
                r = measure(lambda: tri_fa(q, k, v, causal, bm, bn, w, st, gg), attn_flop(B, A, ss, d, causal))
                write(dict(row, **kern_info(kern), **r))
            except Exception as e:
                write(dict(row, error=repr(e)[:300]))


@triton.jit
def k_vec(X, Y, N_ITER: tl.constexpr, BLOCK: tl.constexpr, OP: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x0 = tl.load(X + offs)
    x1 = x0 + 0.1
    x2 = x0 + 0.2
    x3 = x0 + 0.3
    for i in range(N_ITER):
        if OP == 0:  # FMA
            x0 = x0 * 0.999 + 0.0001
            x1 = x1 * 0.999 + 0.0001
            x2 = x2 * 0.999 + 0.0001
            x3 = x3 * 0.999 + 0.0001
        elif OP == 1:  # exp2
            x0 = tl.math.exp2(x0) * 0.25
            x1 = tl.math.exp2(x1) * 0.25
            x2 = tl.math.exp2(x2) * 0.25
            x3 = tl.math.exp2(x3) * 0.25
        else:  # exp
            x0 = tl.exp(x0) * 0.25
            x1 = tl.exp(x1) * 0.25
            x2 = tl.exp(x2) * 0.25
            x3 = tl.exp(x3) * 0.25
    tl.store(Y + offs, x0 + x1 + x2 + x3)


@triton.jit
def k_dot(A, B, C, N_ITER: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, RELOAD: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = tl.load(A + rm[:, None] * BK + rk[None, :])
    b = tl.load(B + rk[:, None] * BN + rn[None, :])
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for i in range(N_ITER):
        if RELOAD:  # re-read operands from L1
            a = tl.load(A + (i % 2) * BM * BK + rm[:, None] * BK + rk[None, :])
            b = tl.load(B + (i % 2) * BK * BN + rk[:, None] * BN + rn[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(C + tl.program_id(0) * BM * BN + rm[:, None] * BN + rn[None, :], acc)


@triton.jit
def k_attn_core(Q, K, V, O, N_ITER: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr, SOFTMAX: tl.constexpr):
    # FA-2 inner loop with K/V tiles kept in L1
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rd = tl.arange(0, D)
    q = tl.load(Q + rm[:, None] * D + rd[None, :])
    m_i = tl.zeros([BM], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BM], dtype=tl.float32) + 1.0
    acc = tl.zeros([BM, D], dtype=tl.float32)
    for i in range(N_ITER):
        k = tl.load(K + (i % 2) * BN * D + rn[None, :] * D + rd[:, None])  # K^T
        v = tl.load(V + (i % 2) * BN * D + rn[:, None] * D + rd[None, :])
        qk = tl.dot(q, k)
        if SOFTMAX:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * 0.125)
            qk = qk * 0.125 - m_ij[:, None]
            p = tl.math.exp2(qk)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            m_i = m_ij
        else:
            p = qk
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
    acc = acc / l_i[:, None]
    tl.store(O + tl.program_id(0) * BM * D + rm[:, None] * D + rd[None, :], acc.to(tl.bfloat16))


def run_micro():
    nprog = 56 * 64
    # vector rates
    BLOCK, w = 1024, 8
    NI = 2000 if args.smoke else 20000
    X = torch.rand(nprog * BLOCK, device=dev, dtype=torch.float32); Y = torch.empty_like(X)
    for op, name in [(0, "fma"), (1, "exp2"), (2, "exp")]:
        fn = lambda: k_vec[(nprog,)](X, Y, N_ITER=NI, BLOCK=BLOCK, OP=op, num_warps=w)
        n_elem_iter = 4.0 * nprog * BLOCK * NI
        r = measure(fn, 2.0 * n_elem_iter if op == 0 else n_elem_iter)  # tflops = Tops/s
        write(dict(kind="vec", tag=name, num_warps=w, programs=nprog, n_iter=NI, ops_per_launch=n_elem_iter, **r))
    # DPAS rates
    NI = 200 if args.smoke else 4000
    cfgs = [(64, 64, 32, 4), (128, 64, 32, 8), (128, 128, 32, 8), (128, 128, 64, 16), (256, 128, 32, 16), (128, 64, 128, 8)]
    for BM, BN, BK, w in (cfgs[:2] if args.smoke else cfgs):
        A = torch.randn(2 * BM * BK, device=dev, dtype=bf16); Bm = torch.randn(2 * BK * BN, device=dev, dtype=bf16)
        C = torch.empty(nprog * BM * BN, device=dev, dtype=torch.float32)
        for reload in (0, 1):
            for g in ("default", "256"):
                row = dict(kind="dot", BM=BM, BN=BN, BK=BK, num_warps=w, reload=reload, grf_mode=g, programs=nprog, n_iter=NI)
                try:
                    fn = lambda: k_dot[(nprog,)](A, Bm, C, N_ITER=NI, BM=BM, BN=BN, BK=BK, RELOAD=reload, num_warps=w, grf_mode=g)
                    kern = fn(); torch.xpu.synchronize()
                    r = measure(fn, 2.0 * BM * BN * BK * NI * nprog)
                    write(dict(row, **kern_info(kern), **r))
                except Exception as e:
                    write(dict(row, error=repr(e)[:300]))
    # attention inner loop with L1-resident K/V, with and without softmax
    NI = 100 if args.smoke else 2000
    for D, BM, BN, w in ([(64, 128, 64, 8)] if args.smoke else
                         [(32, 128, 32, 8), (64, 128, 64, 8), (128, 128, 64, 8), (128, 128, 64, 16), (256, 64, 64, 8), (256, 128, 64, 16)]):
        Q = torch.randn(BM * D, device=dev, dtype=bf16); K = torch.randn(2 * BN * D, device=dev, dtype=bf16)
        V = torch.randn(2 * BN * D, device=dev, dtype=bf16); O = torch.empty(nprog * BM * D, device=dev, dtype=bf16)
        for sm in (1, 0):
            for g in ("default", "256"):
                row = dict(kind="attn_core", d=D, BM=BM, BN=BN, num_warps=w, softmax=sm, grf_mode=g, programs=nprog, n_iter=NI)
                try:
                    fn = lambda: k_attn_core[(nprog,)](Q, K, V, O, N_ITER=NI, BM=BM, BN=BN, D=D, SOFTMAX=sm, num_warps=w, grf_mode=g)
                    kern = fn(); torch.xpu.synchronize()
                    r = measure(fn, 4.0 * BM * BN * D * NI * nprog)
                    write(dict(row, **kern_info(kern), **r))
                except Exception as e:
                    write(dict(row, error=repr(e)[:300]))


hdr = dict(started=time.strftime("%Y-%m-%dT%H:%M:%S"), host=host, sub=args.sub, smoke=args.smoke, torch=torch.__version__,
           triton=triton.__version__, device=str(torch.xpu.get_device_properties(dev)),
           env={k: v for k, v in os.environ.items() if k.startswith(("ZE_", "ONEDNN", "TRITON", "PBS_JOBID", "SYCL"))})
with open(args.out + f"_{args.sub}_header.json", "a") as f:
    f.write(json.dumps(hdr) + "\n")
gemm_ceiling("start")
try:
    {"sdpa": run_sdpa, "ringx": run_ringx, "tricfg": run_tricfg, "micro": run_micro}[args.sub]()
except Exception:
    traceback.print_exc()
gemm_ceiling("end", seconds=0.5)
print("rows", rows_written[0], flush=True)
