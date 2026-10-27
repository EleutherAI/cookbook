#!/usr/bin/env python3
"""Fit one throughput constant per kernel to the aligned K sweep using the replay's wave and padding efficiency."""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "replay"))
from gemm_predict import predict

pts = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data", "e9", "e9_fig2_points.csv")
d = pd.read_csv(pts)
d = d[(d.k % 32 == 0) & (((d.k >= 5000) & (d.k <= 24000)) | (d.k >= 41000))]
rows = []
for r in d.itertuples():
    p = predict(2048, int(r.k), 2048)
    rows.append(dict(k=r.k, tflops=r.tflops_median, entry=p["entry"], kv=bool(p["kv"]), wave_eff=p["wave_eff"], pad_eff=p["pad_eff"]))
x = pd.DataFrame(rows)
x["k_range"] = np.where(x.k >= 41000, ">=41000", "5000-24000")
x["eff"] = np.where(x.kv, x.pad_eff, x.wave_eff * x.pad_eff)
x["norm"] = x.tflops / x.eff
P = x.groupby(["k_range", "entry", "kv"]).norm.median()
x["pred"] = [P[key] * e for key, e in zip(zip(x.k_range, x.entry, x.kv), x.eff)]
x["abs_err"] = (x.pred - x.tflops).abs() / x.tflops
pd.set_option("display.width", 120)
print(x.groupby(["k_range", "entry", "kv"]).agg(points=("k", "size"), P_c=("norm", "median"), mean_abs_err=("abs_err", "mean")).round(3))
print("\noverall mean absolute error per k range:")
print(x.groupby("k_range").abs_err.agg(points="size", mean_abs_err="mean").round(3))
