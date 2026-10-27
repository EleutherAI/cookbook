"""Attention throughput and fraction of XMX peak against head dimension (data/attn)."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
R = f"{HERE}/../data/attn"
OUT = f"{HERE}/out"
__import__("os").makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
cmap = plt.get_cmap("Dark2")
d = pd.read_csv(f"{R}/attn_all.csv")

sd = d[(d.kind == "sdpa") & (d.tag == "d_sweep")].sort_values("d")
tri = d[d.kind == "tri_fa"].sort_values("tflops").groupby("d").tail(1).sort_values("d")
rx = d[(d.kind == "ringx_with_permutes") & (d.causal == 0) & (d.s == 2048)].sort_values("d")
gemm = d[d.kind == "gemm_ceiling"]
g_tf, g_frac = gemm.tflops.median(), gemm.frac_of_xmx_at_clock.median()

fig, (a, b) = plt.subplots(2, 1, figsize=(4.8, 5.4), sharex=True)
a.axhline(g_tf, color="0.4", ls="--", lw=1)
a.text(17, g_tf - 4, f"GEMM, same process ({g_tf:.0f} TFLOP/s)", va="top", fontsize=9, color="0.3")
a.plot(sd.d, sd.tflops, "o-", color=cmap(0), ms=4, label="SDPA (oneDNN)")
a.plot(tri.d, tri.tflops, "s-", color=cmap(1), ms=5, label="Triton FA-2, best tile")
a.plot(rx.d, rx.tflops, "^:", color=cmap(2), ms=5, label="Triton FA-2 via RingX")
a.set_ylabel("Throughput (TFLOP/s)")
a.set_ylim(0, 200)
a.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 0.78))

# fraction of XMX peak at the measured clock
model = {32: 0.20, 64: 0.28}  # no-overlap model, notes/attention.md
nosm = {32: 0.617, 64: 0.562}  # inner loop without softmax
b.axhline(g_frac, color="0.4", ls="--", lw=1)
b.text(17, g_frac - 0.02, f"GEMM ({g_frac:.2f})", va="top", fontsize=9, color="0.3")
b.plot(sd.d, sd.frac_of_xmx_at_clock, "o-", color=cmap(0), ms=4, label="SDPA")
b.plot(tri.d, tri.frac_of_xmx_at_clock, "s-", color=cmap(1), ms=5, label="Triton FA-2, best tile")
b.plot(list(nosm), list(nosm.values()), "D", color=cmap(3), ms=5, label="FA-2 inner loop without softmax")
b.plot(list(model), list(model.values()), "x--", color="k", ms=7, lw=1, label="No-overlap model")
b.set_ylabel("Fraction of XMX peak at clock")
b.set_ylim(0, 1.0)
b.set_xscale("log", base=2)
b.set_xticks([16, 32, 64, 128, 256, 512])
b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
b.set_xlabel("Head dimension $d_h$")
b.legend(fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, 0.9))
for ax, l in zip((a, b), "ab"):
    ax.text(0.015, 0.97, f"({l})", transform=ax.transAxes, va="top")
fig.tight_layout()
fig.savefig(f"{OUT}/attention_dsweep.png", dpi=300)
print(sd[["d", "tflops", "MHz", "frac_of_xmx_at_clock"]].to_string())
print(tri[["d", "tflops", "MHz", "frac_of_xmx_at_clock", "BLOCK_M", "BLOCK_N", "num_warps"]].to_string())
print(rx[["d", "tflops"]].to_string(), g_tf, g_frac)
