#!/usr/bin/env python3
"""Flag points of the original sweeps whose median falls in the post-idle boost. Usage: ORIG_CSV=<dir> python3 analysis/e11a_boost_flags.py <outdir>"""
import sys, glob, os, pandas as pd, numpy as np
D = os.environ["ORIG_CSV"]
OUT = sys.argv[1]
W = 0.35; NW, NM = 50, 100
def flops(col, r):
    b, s, h, a, v = r.train_micro_batch_size_per_gpu, r.seq_length, r.hidden_size, r.num_attention_heads, r.vocab_size
    d = h // a
    return {"qkv_transform": 2*s*b*h*3*h, "attention_score": 2*b*a*s*d*s, "attention_over_value": 2*b*a*s*s*d,
            "attention_linear_projection": 2*s*b*h*h, "mlp_h_to_4h": 2*s*b*h*4*h, "mlp_4h_to_h": 2*s*b*4*h*h,
            "logit_block": 2*s*b*v*h, "sdpa": 4*b*a*s*s*d, "flash_attn": 4*b*s*s*a*d}[col]
GEMMS = ["qkv_transform", "attention_score", "attention_over_value", "attention_linear_projection", "mlp_h_to_4h",
         "mlp_4h_to_h", "logit_block", "sdpa", "flash_attn"]
rows = []
for f in sorted(glob.glob(f"{D}/results_*/*.csv")):
    sweep = os.path.basename(os.path.dirname(f)); fn = os.path.basename(f)
    try: df = pd.read_csv(f)
    except Exception: continue
    if df.empty: continue
    df = df.drop_duplicates(keep="last")
    for r in df.itertuples():
        if "actual_duration" in df.columns:          # full-layer sweeps: whole layer timed as one call
            t = r.actual_duration
            if pd.notna(t) and t > 0:
                rows.append(dict(sweep=sweep, file=fn, a=r.num_attention_heads, h=r.hidden_size, s=r.seq_length, gemm="layer",
                                 tflops=r.actual_throughput, t_call_s=t))
            continue
        for c in GEMMS:
            if c in df.columns:
                tf = getattr(r, c)
                if pd.notna(tf) and tf > 0:
                    rows.append(dict(sweep=sweep, file=fn, a=r.num_attention_heads, h=r.hidden_size, s=r.seq_length, gemm=c,
                                     tflops=tf, t_call_s=flops(c, r) / (tf * 1e12)))
x = pd.DataFrame(rows)
x["run_s"] = (NW + NM) * x.t_call_s
x["f_boost"] = ((W - NW * x.t_call_s) / (NM * x.t_call_s)).clip(0, 1)
x["flag_run_lt_0p5s"] = x.run_s < 0.5
x["class"] = np.select([x.f_boost >= 0.5, x.f_boost > 0], ["median_in_boost", "partial"], "steady")
x.to_csv(f"{OUT}/e11a_points_flagged.csv", index=False)
print("points:", len(x), " by sweep:", x.sweep.value_counts().to_dict())
print("\nshare with run < 0.5 s, and boost class, by sweep x gemm:")
t = x.groupby(["sweep", "gemm"]).agg(n=("run_s", "size"), run_lt_0p5=("flag_run_lt_0p5s", "mean"),
        median_in_boost=("class", lambda c: (c == "median_in_boost").mean()), partial=("class", lambda c: (c == "partial").mean()),
        run_s_median=("run_s", "median"), tflops_median=("tflops", "median"))
print(t.round(3).to_string())
print("\noverall: run<0.5 s %.1f%%, median_in_boost %.1f%%, partial %.1f%%" % (100*x.flag_run_lt_0p5s.mean(),
      100*(x["class"]=="median_in_boost").mean(), 100*(x["class"]=="partial").mean()))
# where the high-throughput points sit
hi = x[x.tflops > 185]
print("points > 185 TFLOP/s: %d, of which run < 0.5 s: %d" % (len(hi), hi.flag_run_lt_0p5s.sum()))
