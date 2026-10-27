#!/usr/bin/env python3
"""Join a gemm_bench.py CSV with the selector replay. Usage: python3 analysis/compare_replay.py <bench.csv> [--out joined.csv]"""
import sys, os, argparse
import pandas as pd, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "replay"))
from gemm_predict import predict

ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--out"); a = ap.parse_args()
df = pd.read_csv(a.csv)
rows = []
for r in df.itertuples():
    p = predict(int(r.m), int(r.k), int(r.n), transA_t=(r.layout == "TN"), transB_t=(r.layout == "NT"), batch=int(r.batch))
    rows.append(dict(pred_copy_based=p["copy_based"], pred_entry=p.get("entry"), pred_tile=f"{p.get('Tm')}x{p.get('Tn')}" if not p["copy_based"] else "copy",
                     pred_kv=p.get("kv"), pred_waves=(p["W"] / p["slots"]) if not p["copy_based"] else np.nan,
                     pred_pad_eff=p.get("pad_eff"), pred_wave_eff=p.get("wave_eff"), pred_eff=p.get("eff"),
                     pred_align_A=p["prob"]["align"][0], pred_align_B=p["prob"]["align"][1], pred_align_C=p["prob"]["align"][2]))
j = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
j["v2_k"] = [(int(x) & -int(x)).bit_length() - 1 for x in j.k]
out = a.out or a.csv.replace(".csv", ".joined.csv"); j.to_csv(out, index=False); print("wrote", out)
print("\nmedian TFLOP/s (min-time based) by predicted configuration:")
print(j.groupby(["pred_copy_based", "pred_entry", "pred_kv"], dropna=False).tflops_min.agg(["count", "median", "min", "max"]).round(1))
# square-wave check for k = 2 mod 4 at fixed m=n=2048, NN
o = j[(j.layout == "NN") & (j.k % 4 == 2) & (j.pred_entry == 463)]
if len(o) > 20:
    lo, hi = o[o.pred_kv].tflops_min.median(), o[~o.pred_kv.astype(bool)].tflops_min.median()
    thr = (lo + hi) / 2
    agree = ((o.tflops_min < thr) == o.pred_kv.astype(bool)).mean()
    print(f"\nk = 2 mod 4, entry 463: KV-level median {lo:.1f}, no-KV median {hi:.1f}, threshold {thr:.1f}; agreement {agree*100:.2f}% of {len(o)} points")
