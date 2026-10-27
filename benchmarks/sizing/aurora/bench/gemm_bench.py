#!/usr/bin/env python3
"""bf16 GEMM benchmark on one stack. C[m x k] = A[m x n] @ B[n x k]; NN = mm, NT = F.linear forward, TN = weight grad."""
import argparse, csv, json, math, os, socket, sys, time
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--m", type=int, nargs="+", default=[2048])
p.add_argument("--n", type=int, nargs="+", default=[2048])
p.add_argument("--k", type=int, nargs="+")
p.add_argument("--k_range", type=int, nargs=3)
p.add_argument("--shapes", help="file with lines 'm n k'")
p.add_argument("--layout", choices=["NN", "NT", "TN"], default="NN")
p.add_argument("--batch", type=int, default=1)
p.add_argument("--iters", type=int, default=200)
p.add_argument("--warmup", type=int, default=50)
p.add_argument("--rotate_mb", type=float, default=0, help="cycle operand copies until they exceed this size")
p.add_argument("--burst", type=int, default=1, help="calls between timing events")
p.add_argument("--init", choices=["device", "host"], default="device")
p.add_argument("--idle_s", type=float, default=0)
p.add_argument("--dump_iters", action="store_true")
p.add_argument("--profile", action="store_true")
p.add_argument("--out", required=True)
p.add_argument("--tag", default="")
a = p.parse_args()

dev = torch.device("xpu", 0)
if a.shapes:
    shapes = [tuple(map(int, l.split()[:3])) for l in open(a.shapes) if l.strip() and not l.startswith("#")]
else:
    shapes = [(m, n, k) for m in a.m for n in a.n for k in (a.k or range(*a.k_range))]

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out + ".header.json", "a") as f:
    f.write(json.dumps(dict(started=time.strftime("%Y-%m-%dT%H:%M:%S"), host=socket.gethostname(), argv=sys.argv,
                            torch=torch.__version__, device=str(torch.xpu.get_device_properties(dev)))) + "\n")

FIELDS = ["tag", "layout", "batch", "m", "n", "k", "rotate", "burst", "init", "footprint_MB", "l2_resident_guess",
          "iters", "warmup", "min_ms", "p10_ms", "median_ms", "p90_ms", "max_ms", "mean_ms", "warmup_min_ms",
          "first_ms", "tflops_min", "tflops_median", "tflops_p10time", "host_t0", "host_t1"]
new = not os.path.exists(a.out)
w = csv.DictWriter(fout := open(a.out, "a", newline=""), fieldnames=FIELDS)
if new:
    w.writeheader()
if a.dump_iters:
    newi = not os.path.exists(a.out + ".iters.csv")
    wi = csv.writer(fit := open(a.out + ".iters.csv", "a", newline=""))
    if newi:
        wi.writerow(["tag", "layout", "batch", "m", "n", "k", "i", "is_warmup", "host_start", "ms"])


def make(*shape):
    if a.init == "device":
        return torch.randn(shape, device=dev, dtype=torch.bfloat16)
    return torch.randn(shape).bfloat16().to(dev)


def operands(m, n, k):
    pre = (a.batch,) if a.batch > 1 else ()
    A = make(*pre, n, m).transpose(-1, -2) if a.layout == "TN" else make(*pre, m, n)
    B = make(*pre, k, n).transpose(-1, -2) if a.layout == "NT" else make(*pre, n, k)
    return A, B, torch.empty(pre + (m, k), device=dev, dtype=torch.bfloat16)


def run(A, B, C):
    (torch.bmm if a.batch > 1 else torch.mm)(A, B, out=C)


st, en = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
for m, n, k in shapes:
    b = a.batch
    fp = 2 * b * (m * n + n * k + m * k)
    R = max(1, min(math.ceil(a.rotate_mb * 1e6 / fp), int(40e9 // fp)))
    time.sleep(a.idle_s)
    sets = [operands(m, n, k) for _ in range(R)]
    torch.xpu.synchronize()
    N = a.warmup + a.iters
    times, hst = np.zeros(N), np.zeros(N)
    with torch.no_grad():
        for i in range(N):
            hst[i] = time.time()
            st.record()
            for _ in range(a.burst):
                run(*sets[i % R])
            en.record()
            torch.xpu.synchronize()
            times[i] = st.elapsed_time(en) / a.burst
        if a.profile:
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True) as prof:
                for i in range(5):
                    run(*sets[i % R])
                torch.xpu.synchronize()
            prof.export_chrome_trace(f"{a.out}.trace_{a.layout}_{b}x{m}x{n}x{k}.json")
            print(prof.key_averages().table(sort_by="self_xpu_time_total", row_limit=15), flush=True)
    t = times[a.warmup:]
    tf = lambda ms: 2.0 * b * m * n * k / (ms * 1e-3) / 1e12
    row = dict(tag=a.tag, layout=a.layout, batch=b, m=m, n=n, k=k, rotate=R, burst=a.burst, init=a.init,
               footprint_MB=round(fp / 1e6, 2), l2_resident_guess=int(fp * R < 192 * 2**20), iters=a.iters, warmup=a.warmup,
               min_ms=t.min(), p10_ms=np.percentile(t, 10), median_ms=np.median(t), p90_ms=np.percentile(t, 90),
               max_ms=t.max(), mean_ms=t.mean(), warmup_min_ms=times[:a.warmup].min() if a.warmup else float("nan"),
               first_ms=times[0], tflops_min=tf(t.min()), tflops_median=tf(np.median(t)),
               tflops_p10time=tf(np.percentile(t, 10)), host_t0=hst[0], host_t1=hst[-1])
    w.writerow(row)
    fout.flush()
    if a.dump_iters:
        for i in range(N):
            wi.writerow([a.tag, a.layout, b, m, n, k, i, int(i < a.warmup), f"{hst[i]:.6f}", f"{times[i]:.6f}"])
        fit.flush()
    print(f"{a.layout} b={b} {m}x{n}x{k} R={R}: median {row['tflops_median']:.1f} TFLOP/s", flush=True)
    del sets
    torch.xpu.empty_cache()
