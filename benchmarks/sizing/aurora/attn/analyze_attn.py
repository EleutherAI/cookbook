#!/usr/bin/env python3
"""Join attn_bench.py rows with xpu-smi clock and power. Usage: python3 attn/analyze_attn.py <run dir>"""
import sys, os, re, glob, json
import pandas as pd

R = sys.argv[1]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
CORES, XMX = 56, 4096  # Xe-cores per stack, bf16 FLOP/clk/Xe-core
rows = []
for f in sorted(glob.glob(f"{R}/attn_*.jsonl")):
    rows += [json.loads(l) for l in open(f)]
x = pd.DataFrame(rows)
s = pd.read_csv(f"{R}/smi_100ms.txt", skipinitialspace=True)
s.columns = ["ts", "dev", "tile", "util", "W", "MHz", "C", "thr"]
for c in ["W", "MHz", "C"]:
    s[c] = pd.to_numeric(s[c], errors="coerce")
s["t"] = (pd.to_datetime(s.ts) - pd.Timestamp("1970-01-01")).dt.total_seconds()


def clk(r):
    w = s[(s.t >= r.host_t0) & (s.t <= r.host_t1)]
    if len(w) == 0:
        w = s.iloc[(s.t - (r.host_t0 + r.host_t1) / 2).abs().argsort()[:1]]
    return pd.Series(dict(MHz=w.MHz.median(), W=w.W.median(), throttle=w.thr.mode().iloc[0] if len(w) else ""))


ok = x[x.get("error").isna()] if "error" in x else x
ok = ok[ok.host_t0.notna()].copy()
ok[["MHz", "W", "throttle"]] = ok.apply(clk, axis=1)
ok["xmx_peak_at_clock"] = CORES * XMX * ok.MHz * 1e6 / 1e12
ok["frac_of_xmx_at_clock"] = ok.tflops / ok.xmx_peak_at_clock
ok.to_csv(f"{R}/attn_all.csv", index=False)
out = []
P = lambda *a: out.append(" ".join(str(v) for v in a))

P("== GEMM ceiling (torch.mm 2048x2048x16128, same process) per subcommand")
P(ok[ok.kind == "gemm_ceiling"][["sub", "tag", "tflops", "MHz", "W", "frac_of_xmx_at_clock"]].round(3).to_string(index=False))

# ---- vector rates
v = ok[ok.kind == "vec"].set_index("tag")
if len(v):
    fma = v.loc["fma"]; n = fma.ops_per_launch
    P("\n== vector units (fp32, SIMD, 56 Xe-cores)")
    fma_clk = 2 * n / (fma.median_ms * 1e-3) / (CORES * fma.MHz * 1e6)
    P(f"FMA: {fma.tflops:.2f} TFLOP/s at {fma.MHz:.0f} MHz = {fma_clk:.0f} FLOP/clk/Xe-core (spec 256)")
    for t in ("exp2", "exp"):
        if t in v.index:
            e = v.loc[t]
            # exp cycles = exp-kernel cycles - FMA-kernel cycles
            cyc_ex = e.median_ms * 1e-3 * e.MHz * 1e6 - fma.median_ms * 1e-3 * fma.MHz * 1e6
            per_clk = n / cyc_ex / CORES
            raw = n / (e.median_ms * 1e-3) / (CORES * e.MHz * 1e6)
            P(f"{t}: kernel {e.median_ms:.2f} ms vs FMA kernel {fma.median_ms:.2f} ms at {e.MHz:.0f} MHz -> "
              f"{per_clk:.1f} {t}/clk/Xe-core after subtracting the mul ({raw:.1f} incl. mul)")

# ---- DPAS
d = ok[ok.kind == "dot"]
if len(d):
    d = d.assign(flop_clk_core=d.tflops * 1e12 / (CORES * d.MHz * 1e6))
    P("\n== DPAS (tl.dot loop; reload=1: operands re-read from L1 every step)")
    P(d[["BM", "BN", "BK", "num_warps", "reload", "grf_mode", "tflops", "MHz", "W", "flop_clk_core", "frac_of_xmx_at_clock", "n_spills"]].round(3).to_string(index=False))

c = ok[ok.kind == "attn_core"]
if len(c):
    c = c.assign(flop_clk_core=c.tflops * 1e12 / (CORES * c.MHz * 1e6))
    P("\n== FA-2 inner loop, K/V L1-resident (softmax=0: two dots + bf16 convert only)")
    P(c[["d", "BM", "BN", "num_warps", "softmax", "grf_mode", "tflops", "MHz", "W", "flop_clk_core", "frac_of_xmx_at_clock", "n_regs", "n_spills"]].round(3).to_string(index=False))

# ---- Triton config sweep
t = ok[ok.kind == "tri_fa"]
if len(t):
    P("\n== Triton FA-2 forward (ringX kernel), b=4 a=32 s=2048 non-causal, config sweep: top 5 per d")
    cols = ["d", "BLOCK_M", "BLOCK_N", "num_warps", "num_stages", "grf_mode", "tflops", "MHz", "frac_of_xmx_at_clock", "n_regs", "n_spills", "grf_256"]
    for dd, g in t.groupby("d"):
        P(g.sort_values("tflops", ascending=False)[cols].head(5).round(3).to_string(index=False))
    P("\nspill vs throughput (all configs):")
    P(t.assign(spills=t.n_spills > 0).groupby(["d", "spills"]).tflops.agg(["count", "median", "max"]).round(1).to_string())
    P("\nby num_warps (median over tiles):")
    P(t.groupby(["d", "num_warps"]).tflops.agg(["median", "max"]).unstack().round(1).to_string())
    err = x[(x.get("kind") == "tri_fa") & x.get("error").notna()] if "error" in x else pd.DataFrame()
    P(f"failed configs: {len(err)}" + ("" if not len(err) else " e.g. " + str(err.iloc[0].error)[:150]))
fu = ok[ok.kind == "tri_fa_followup"]
if len(fu):
    P("\n== follow-ups on the best tile per d (stages, forced 128-GRF, causal, s=8192)")
    P(fu[["d", "s", "causal", "BLOCK_M", "BLOCK_N", "num_warps", "num_stages", "grf_mode", "tflops", "MHz", "n_regs", "n_spills"]].round(2).to_string(index=False))

# ---- SDPA
sd = ok[ok.kind == "sdpa"]
if len(sd):
    P("\n== SDPA (torch 2.10 -> oneDNN graph sdp partition), b=4")
    P(sd[["tag", "a", "s", "d", "causal", "tflops", "MHz", "W", "frac_of_xmx_at_clock"]].round(3).to_string(index=False))

# ---- ringX autotuned
rx = ok[ok.kind.astype(str).str.startswith("ringx")]
if len(rx):
    P("\n== sizing-benchmark path: ringX fused (own autotuner, num_stages/warps/BLOCK as chosen)")
    P(rx[["kind", "d", "s", "causal", "tflops", "MHz", "autotune_s", "best_config"]].round(2).to_string(index=False))

# ---- kernel identification
ids = []
for f in sorted(glob.glob(f"{R}/id_*.log")):
    txt = open(f, errors="replace").read()
    impl = sorted(set(re.findall(r"onednn_verbose,v1,graph,exec,gpu,(\d+),(\w+),([^,]+),", txt)))
    props = re.findall(r'^\s*"?([A-Za-z_][^"\[]*)\[SIMD(\d+) \{([^}]*)\} \{([^}]*)\}\]"?,\s*(AOT|JIT),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)', txt, re.M)
    regs = re.findall(r"n_regs=(\S+) n_spills=(\S+) threads_per_warp=(\S+) shared=(\S+)", txt)
    for name, simd, grid, wg, comp, simd2, nargs, slm, priv, spill in props:
        if "Distribution" in name or "Fill" in name or "Elementwise" in name:
            continue
        ids.append(dict(log=os.path.basename(f), kernel=name.strip(), simd=simd, grid=grid, wg=wg, slm_B=int(slm), spill_B=int(spill),
                        onednn=";".join(f"{p}:{o}" for _, p, o in impl)[:120], triton_regs=";".join("/".join(r) for r in regs)))
if ids:
    I = pd.DataFrame(ids).drop_duplicates()
    I.to_csv(f"{R}/attn_ids.csv", index=False)
    P("\n== kernel identification (unitrace -d -v, ONEDNN_VERBOSE graph partitions)")
    P(I.to_string(index=False))
open(f"{R}/attn_summary.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
