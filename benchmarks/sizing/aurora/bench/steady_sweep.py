#!/usr/bin/env python3
"""Steady-state torch.mm sweep on one stack: pre-heat, operands allocated once, anchor shapes at the start and end."""
import argparse, csv, json, os, socket, sys, time
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--points", required=True, help="file with lines 'm n k'")
p.add_argument("--out", required=True, help="output prefix")
p.add_argument("--tag", default="")
p.add_argument("--warmup", type=int, default=50)
p.add_argument("--iters", type=int, default=200)
p.add_argument("--preheat_s", type=float, default=2.0)
p.add_argument("--preheat_shape", type=int, nargs=3, default=[2048, 2048, 16128])
p.add_argument("--anchors", type=int, nargs="*", default=[8192, 16384, 32768, 65536, 98560, 98561, 98562, 98563])
p.add_argument("--budget_s", type=float, default=1e9, help="stop starting new points after this many seconds")
a = p.parse_args()

t_start = time.time()
dev = torch.device("xpu", 0)
pts = [tuple(map(int, l.split()[:3])) for l in open(a.points) if l.strip() and not l.startswith("#")]
anchors = [(2048, 2048, k) for k in a.anchors]
allp = pts + anchors + [tuple(a.preheat_shape)]


def buffer(n, fill=True):
    buf = torch.empty(n, device=dev, dtype=torch.bfloat16)
    if fill:
        for s in range(0, n, 1 << 28):
            buf[s:s + (1 << 28)].normal_()
    return buf


bufA = buffer(max(m * n for m, n, k in allp))
bufB = buffer(max(n * k for m, n, k in allp))
bufC = buffer(max(m * k for m, n, k in allp), fill=False)
torch.xpu.synchronize()
host = socket.gethostname()
with open(a.out + "_header.json", "a") as f:
    f.write(json.dumps(dict(started=time.strftime("%Y-%m-%dT%H:%M:%S"), host=host, argv=sys.argv, torch=torch.__version__,
                            pbs_jobid=os.environ.get("PBS_JOBID"), n_points=len(pts))) + "\n")

FIELDS = ["tag", "phase", "host", "m", "n", "k", "v2_k", "iters", "warmup", "min_ms", "p10_ms", "median_ms", "p90_ms", "max_ms",
          "mean_ms", "warmup_median_ms", "first_ms", "tflops_median", "tflops_p10time", "tflops_p90time", "tflops_min",
          "gap_before_ms", "host_t0", "host_t1"]
full, med = a.out + "_full.csv", a.out + "_median.csv"
newf, newm = not os.path.exists(full), not os.path.exists(med)
wf = csv.DictWriter(ff := open(full, "a", newline=""), fieldnames=FIELDS)
wm = csv.writer(fm := open(med, "a", newline=""))
if newf:
    wf.writeheader()
if newm:
    wm.writerow(["m", "n", "k", "throughput"])
st, en = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
last_end = None


def views(m, n, k):
    return bufA[:m * n].view(m, n), bufB[:n * k].view(n, k), bufC[:m * k].view(m, k)


def preheat():
    global last_end
    A, B, C = views(*a.preheat_shape)
    t0 = time.time()
    while time.time() - t0 < a.preheat_s:
        for _ in range(10):
            torch.mm(A, B, out=C)
        torch.xpu.synchronize()
    last_end = time.time()


def measure(m, n, k, phase):
    global last_end
    A, B, C = views(m, n, k)
    N = a.warmup + a.iters
    times = np.zeros(N)
    t0 = time.time()
    gap = (t0 - last_end) * 1e3
    with torch.no_grad():
        for i in range(N):
            st.record()
            torch.mm(A, B, out=C)
            en.record()
            torch.xpu.synchronize()
            times[i] = st.elapsed_time(en)
    last_end = time.time()
    t = times[a.warmup:]
    tf = lambda ms: 2.0 * m * n * k / (ms * 1e-3) / 1e12
    row = dict(tag=a.tag, phase=phase, host=host, m=m, n=n, k=k, v2_k=(k & -k).bit_length() - 1, iters=a.iters,
               warmup=a.warmup, min_ms=t.min(), p10_ms=np.percentile(t, 10), median_ms=np.median(t),
               p90_ms=np.percentile(t, 90), max_ms=t.max(), mean_ms=t.mean(),
               warmup_median_ms=np.median(times[:a.warmup]) if a.warmup else float("nan"), first_ms=times[0],
               tflops_median=tf(np.median(t)), tflops_p10time=tf(np.percentile(t, 10)),
               tflops_p90time=tf(np.percentile(t, 90)), tflops_min=tf(t.min()), gap_before_ms=gap, host_t0=t0, host_t1=last_end)
    wf.writerow(row)
    ff.flush()
    if phase == "point":
        wm.writerow([m, n, k, f"{row['tflops_median']:.3f}"])
        fm.flush()
    return row


status = dict(done=0, total=len(pts), last=None, stop="complete")
preheat()
for s in anchors:
    measure(*s, "anchor_start")
for i, s in enumerate(pts):
    if time.time() - t_start > a.budget_s:
        status["stop"] = "budget"
        break
    r = measure(*s, "point")
    status.update(done=i + 1, last=list(s))
    if i % 500 == 0:
        print(f"[{time.time() - t_start:.0f}s] {i + 1}/{len(pts)} {s}: {r['tflops_median']:.1f} TFLOP/s", flush=True)
for s in anchors:
    measure(*s, "anchor_end")
status["elapsed_s"] = round(time.time() - t_start, 1)
json.dump(status, open(a.out + "_status.json", "w"))
print(status, flush=True)
