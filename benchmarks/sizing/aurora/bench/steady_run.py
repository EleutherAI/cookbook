#!/usr/bin/env python3
"""Back-to-back torch.mm on one stack for --duration s from a common start time, for concurrent runs on several stacks."""
import argparse, json, os, socket, time
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--k", type=int, required=True)
p.add_argument("--duration", type=float, default=20.0)
p.add_argument("--start_at", type=float, required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
dev = torch.device("xpu", 0)
A = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
B = torch.randn(2048, a.k, device=dev, dtype=torch.bfloat16)
C = torch.empty(2048, a.k, device=dev, dtype=torch.bfloat16)
for _ in range(3):
    torch.mm(A, B, out=C)
torch.xpu.synchronize()
st, en = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
late = time.time() - a.start_at
while time.time() < a.start_at:
    time.sleep(0.001)
hs, ms = [], []
t_end = a.start_at + a.duration
with torch.no_grad():
    while True:
        t = time.time()
        if t >= t_end:
            break
        st.record(); torch.mm(A, B, out=C); en.record(); torch.xpu.synchronize()
        hs.append(t); ms.append(st.elapsed_time(en))
hs, ms = np.array(hs), np.array(ms)
tf = 2 * 2048 * 2048 * a.k / (ms * 1e-3) / 1e12
with open(a.out + "_iters.csv", "w") as f:
    f.write("host_start,ms\n")
    f.writelines(f"{h:.6f},{m:.6f}\n" for h, m in zip(hs, ms))
rel = hs - a.start_at
summ = dict(host=socket.gethostname(), mask=os.environ.get("ZE_AFFINITY_MASK"), k=a.k, calls=len(ms), ready_late_s=round(late, 3),
            tflops_0_0p3s=float(np.median(tf[rel < 0.3])) if (rel < 0.3).any() else None,
            tflops_2s_end=float(np.median(tf[rel >= 2])), tflops_all=float(np.median(tf)),
            tflops_p10t_2s_end=float(np.percentile(tf[rel >= 2], 10)), tflops_p90t_2s_end=float(np.percentile(tf[rel >= 2], 90)))
json.dump(summ, open(a.out + ".json", "w"))
print(json.dumps(summ), flush=True)
