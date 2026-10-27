#!/usr/bin/env python3
"""E10 table: TFLOP/s, clock, power and throttle reason per phase and stack. Usage: python3 analysis/e10_analysis.py <run dir>"""
import sys, glob, json, os, re
import numpy as np, pandas as pd

R = sys.argv[1]
smi = {}
for f in glob.glob(f"{R}/smi_d*_t*.txt"):
    d, t = map(int, re.findall(r"smi_d(\d)_t(\d)", f)[0])
    s = pd.read_csv(f, skipinitialspace=True)
    s.columns = ["ts", "dev", "tile", "util", "W", "MHz", "C", "thr"]
    for c in ["W", "MHz", "C"]:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s["t"] = (pd.to_datetime(s.ts) - pd.Timestamp("1970-01-01")).dt.total_seconds()
    smi[2 * d + t] = s
phases = []
for ln in open(f"{R}/e10_phases.txt"):
    m = re.search(r"phase (\w+) k=(\d+) stacks ([\d ]+) start_at ([\d.]+)", ln)
    phases.append(dict(name=m.group(1), k=int(m.group(2)), stacks=[int(x) for x in m.group(3).split()], t0=float(m.group(4))))
rows = []
for ph in phases:
    for s in range(12):
        sm = smi.get(s)
        w = sm[(sm.t >= ph["t0"] + 2) & (sm.t <= ph["t0"] + 20)] if sm is not None else pd.DataFrame()
        r = dict(phase=ph["name"], k=ph["k"], n_active=len(ph["stacks"]), stack=s, gpu=s // 2, active=s in ph["stacks"],
                 MHz=w.MHz.median() if len(w) else np.nan, W=w.W.median() if len(w) else np.nan, C=w.C.median() if len(w) else np.nan,
                 throttle=w.thr.value_counts().idxmax() if len(w) else "")
        j = f"{R}/{ph['name']}_s{s}.json"
        if os.path.exists(j):
            js = json.load(open(j))
            r.update(TF_0_0p3s=js["tflops_0_0p3s"], TF_steady=js["tflops_2s_end"], calls=js["calls"], late_s=js["ready_late_s"])
        rows.append(r)
x = pd.DataFrame(rows)
x.to_csv(f"{R}/e10_table.csv", index=False)
pd.set_option("display.width", 250)
act = x[x.active]
print("== active stacks: steady TFLOP/s (2-20 s), first 0.3 s, tile MHz / W / temp / throttle")
print(act[["phase", "k", "n_active", "stack", "gpu", "TF_steady", "TF_0_0p3s", "MHz", "W", "C", "throttle", "late_s"]].round(1).to_string(index=False))
print("\n== per phase summary")
g = act.groupby(["phase", "k", "n_active"]).agg(TF_med=("TF_steady", "median"), TF_min=("TF_steady", "min"), TF_max=("TF_steady", "max"),
                                                W_med=("W", "median"), MHz_med=("MHz", "median"))
print(g.round(1).to_string())
print("\n== per-GPU power (sum of both tiles, median over 2-20 s) and idle-tile power")
for ph in phases:
    gp = x[x.phase == ph["name"]].groupby("gpu").W.sum()
    idle = x[(x.phase == ph["name"]) & (~x.active)].W
    print(f"{ph['name']} k={ph['k']} active={ph['stacks']}: GPU W " + " ".join(f"{v:.0f}" for v in gp.values) +
          (f" | idle tiles median {idle.median():.0f} W" if len(idle) else ""))
