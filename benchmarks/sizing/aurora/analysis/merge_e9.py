#!/usr/bin/env python3
"""Merge the E9 chunk outputs in runs/e9_c*/ into the files in data/e9/. Usage: python3 analysis/merge_e9.py <outdir>"""
import glob, os, sys
import pandas as pd

QA = os.environ.get("QA", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)


def load(prefix):
    return pd.concat([pd.read_csv(f).assign(chunk=os.path.basename(os.path.dirname(f)))
                      for f in sorted(glob.glob(f"{QA}/runs/e9_c*/{prefix}_full.csv"))], ignore_index=True)


def points(d):
    return d[d.phase == "point"].drop_duplicates(["m", "n", "k"], keep="last").sort_values("k")


def sizing_csv(d, path):
    d[["m", "n", "k"]].assign(throughput=d.tflops_median.round(3)).to_csv(path, index=False)


k = load("e9k")
pts = points(k)
pts[["chunk", "host", "m", "n", "k", "v2_k", "median_ms", "p10_ms", "p90_ms", "tflops_median", "tflops_p10time",
     "tflops_p90time", "gap_before_ms"]].to_csv(f"{OUT}/e9_fig2_points.csv", index=False, float_format="%.6g")
sizing_csv(pts, f"{OUT}/e9_fig2_basicGemmKSweep.csv")
k[k.phase.str.startswith("anchor")].to_csv(f"{OUT}/e9_anchors.csv", index=False)
mk = points(load("e9mk"))
mk.to_csv(f"{OUT}/e9_fig3_full.csv", index=False)
sizing_csv(mk, f"{OUT}/e9_fig3_basicGemmMKSweep.csv")
