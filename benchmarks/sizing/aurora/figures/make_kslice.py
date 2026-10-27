"""Split-K oscillation over one 1792-wide period (data/e3)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
R = f"{HERE}/../data/e3"
OUT = f"{HERE}/out"
__import__("os").makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})
cmap = plt.get_cmap("Dark2")
CLASSES = [1, 2, 4, 8, 16, 32, 64]
COL = {c: cmap(i) for i, c in enumerate(CLASSES)}

d = pd.read_csv(f"{R}/e3.joined.csv")
d = d[d.tag == "e3_new"].sort_values("k")
cls = np.ones(len(d), int)
for p in [2, 4, 8, 16, 32, 64]:
    cls = np.where(d.k % p == 0, p, cls)
d["cls"] = cls

fig, ax = plt.subplots(figsize=(7.0, 3.6))
k0, k1 = int(d.k.min()), int(d.k.max())
# shade the k where ceil(k/256) mod 7 in {1,2,3}
ks = np.arange(k0, k1 + 1)
on = np.isin(np.ceil(ks / 256).astype(int) % 7, [1, 2, 3])
edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
for a, b in zip(edges[::2], edges[1::2]):
    ax.axvspan(ks[a], ks[b - 1], color="0.88", zorder=0, lw=0)
for x in range(0, k1 + 1792, 1792):
    if k0 <= x <= k1:
        ax.axvline(x, color="0.4", ls=":", lw=1)
labels = {2: "2", 4: "4", 8: "8", 16: "16", 32: "32", 64: "≥64"}
for c in [2, 4, 8, 16, 32, 64]:
    q = d[d.cls == c]
    ax.plot(q.k, q.tflops_median, ".", ms=4 if c == 2 else 3.5, color=COL[c], label=labels[c])
ax.set_xlim(k0, k1)
ax.set_ylim(95, 170)
ax.set_xlabel("k")
ax.set_ylabel("Throughput (TFLOP/s)")
ax.legend(ncol=6, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, 1.27), frameon=False, markerscale=3,
          title="Largest power of 2 dividing k", title_fontsize=9.5)
fig.tight_layout()
fig.savefig(f"{OUT}/kslice_period.png", dpi=300)
print("ok")
