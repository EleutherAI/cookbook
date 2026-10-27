#!/usr/bin/env python3
"""Median unitrace metrics per kernel call, relative to k = 98560. Usage: python3 analysis/e8_parse.py <run dir>"""
import sys, glob, os, re, io
import pandas as pd

R = sys.argv[1]
rows = []
for f in sorted(glob.glob(f"{R}/ut_*_k*.log")):
    g, k = re.findall(r"ut_(.+)_k(\d+)\.log", os.path.basename(f))[0]
    txt = open(f, errors="replace").read()
    if "=== Device #0 Metrics ===" not in txt:
        continue
    block = txt.split("=== Device #0 Metrics ===", 1)[1].split("\n===", 1)[0]   # rows are separated by blank lines
    lines = [ln for ln in block.splitlines() if ln.startswith("Kernel,") or ln.lstrip('"').startswith("gemm_kernel")]
    if len(lines) < 2:
        continue
    d = pd.read_csv(io.StringIO("\n".join(lines)))
    d = d.tail(20)                                 # measured calls (10 warm-up first)
    num = d.select_dtypes("number").median()
    for m, v in num.items():
        rows.append(dict(group=g, k=int(k), metric=m, value=v, n=len(d)))
x = pd.DataFrame(rows)
x.to_csv(f"{R}/e8_metrics_long.csv", index=False)
p = x.pivot_table(index=["group", "metric"], columns="k", values="value")
for k in [c for c in p.columns if c != 98560]:
    p[f"r{k}"] = p[k] / p[98560]
skip = re.compile(r"QueryBeginTime|ContextId|SourceId|StreamMarker|ReportId|ReportsCount|Mid|Report|GlobalInstanceId|SubDeviceId|Overrun|Split|Changed")
p = p[[not skip.search(m) for m in p.index.get_level_values("metric")]]
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_colwidth", 60)
p.to_csv(f"{R}/e8_metrics_table.csv")
print(p.round(3).to_string())
