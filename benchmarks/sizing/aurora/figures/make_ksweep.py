"""K-sweep and M=K figures from data/e9."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
E9 = f"{HERE}/../data/e9"
OUT = f"{HERE}/out"
__import__("os").makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 18, "font.family": "DejaVu Sans"})
DPI = 300
cmap = plt.get_cmap("Dark2")
CLASSES = [1, 2, 4, 8, 16, 32, 64]
COL = {c: cmap(i) for i, c in enumerate(CLASSES)}


def pow2_class(k):
    k = np.asarray(k)
    c = np.ones_like(k)
    for p in [2, 4, 8, 16, 32, 64]:
        c = np.where(k % p == 0, p, c)
    return c


def centred_median(x, y, w):
    x = np.asarray(x); y = np.asarray(y)
    lo = np.searchsorted(x, x - w / 2, "left")
    hi = np.searchsorted(x, x + w / 2, "right")
    return np.array([np.median(y[a:b]) for a, b in zip(lo, hi)])


df = pd.read_csv(f"{E9}/e9_fig2_basicGemmKSweep.csv").sort_values("k")
df["cls"] = pow2_class(df.k.values)


def ksweep(kmax, w, fname, ticks, ylim, vlines=None, legend_loc="lower right"):
    d = df[df.k < kmax]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    if vlines:
        for x in range(vlines, kmax, vlines):
            ax.axvline(x, color="0.6", lw=0.8, ls=":", zorder=0)
    for c in CLASSES:
        q = d[d.cls == c]
        ax.scatter(q.k, q.throughput, s=3, color=COL[c], alpha=0.35, lw=0, rasterized=True)
        ax.plot(q.k, centred_median(q.k, q.throughput, w), color=COL[c], lw=2.2, label=("≥64" if c == 64 else f"{c}"))
    ax.set_xlim(0, kmax)
    ax.set_ylim(*ylim)
    ax.set_xticks(ticks)
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("k")
    ax.set_ylabel("Throughput (TFLOP/s)")
    ax.legend(ncol=4, loc=legend_loc, fontsize=14, handlelength=1.0, columnspacing=0.7, framealpha=0.9,
              title="Largest power of 2 dividing k", title_fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=DPI)
    plt.close(fig)


ksweep(1024, 40, "k_sweep_small.png", range(0, 1025, 128), (0, 130), legend_loc="upper left")
ksweep(16385, 160, "k_sweep_medium.png", range(0, 16385, 2048), (0, 210), vlines=1792)
ksweep(131072, 480, "k_sweep_large.png", range(0, 131073, 16384), (0, 210))

# M=K sweep, n = 4096
f3 = pd.read_csv(f"{E9}/e9_fig3_full.csv").sort_values("k")
fig, ax = plt.subplots(figsize=(6.4, 4.8))
x = f3.k.values
ax.fill_between(x, f3.tflops_p90time, f3.tflops_p10time, color=COL[1], alpha=0.2, lw=0)
boost = (f3.k >= 2048) & (f3.k <= 4096)
ax.plot(x, f3.tflops_median, "-", color=COL[1], lw=2)
ax.plot(x[~boost], f3.tflops_median[~boost], "o", color=COL[1], ms=6)
ax.plot(x[boost], f3.tflops_median[boost], "o", mfc="white", mec=COL[1], mew=2, ms=7)
ax.set_xscale("log", base=2)
ax.set_xticks([2**i for i in range(5, 18, 2)])
ax.set_ylim(0, 250)
ax.set_xlabel("m = k")
ax.set_ylabel("Throughput (TFLOP/s)")
fig.tight_layout()
fig.savefig(f"{OUT}/gemmMKSweep.png", dpi=DPI)
plt.close(fig)
print("ok")
