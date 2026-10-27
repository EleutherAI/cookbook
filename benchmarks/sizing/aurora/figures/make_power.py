"""Throughput, clock, power and temperature from idle to sustained (data/e4c)."""
import numpy as np, pandas as pd, matplotlib, datetime as dt
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
R = f"{HERE}/../data/e4c"
OUT = f"{HERE}/out"
__import__("os").makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})
cmap = plt.get_cmap("Dark2")

it = pd.read_csv(f"{R}/e4c.csv.iters.csv")
smi = pd.read_csv(f"{R}/e4c_smi_100ms.txt", skipinitialspace=True)
smi.columns = [c.strip() for c in smi.columns]
smi["t"] = [dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp() for s in smi["Timestamp"]]
for c in ["GPU Power (W)", "GPU Frequency (MHz)", "GPU Core Temperature (Celsius Degree)"]:
    smi[c] = pd.to_numeric(smi[c], errors="coerce")

runs = [("long_reuse_16128", "k = 16128, operands reused", cmap(0)),
        ("long_rot800_16128", "k = 16128, operands rotated over 800 MB", cmap(1)),
        ("long_reuse_50176", "k = 50176, operands reused", cmap(2))]

fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
(a, b), (c, d) = axs
bins = np.logspace(-2.3, np.log10(20), 70)
for tag, lab, col in runs:
    q = it[it.tag == tag].copy()
    t0 = q.host_start.iloc[0]
    q["t"] = q.host_start - t0 + q.ms / 1e3
    q["tf"] = 2 * q.m * q.n * q.k / (q.ms * 1e-3) / 1e12
    q = q[q.i > 0]  # call 0 includes JIT
    idx = np.digitize(q.t, bins)
    g = q.groupby(idx)
    tt = g.t.median(); tf = g.tf.median()
    a.plot(tt, tf, "-", color=col, lw=1.8, label=lab)
    t1 = q.t.max() + t0
    s = smi[(smi.t >= t0) & (smi.t <= t1)]
    s = s[pd.to_numeric(s["GPU Utilization (%)"], errors="coerce") > 50]
    ts = s.t - t0
    b.plot(ts, s["GPU Frequency (MHz)"], "-", color=col, lw=1.4)
    c.plot(ts, s["GPU Power (W)"], "-", color=col, lw=1.4)
    d.plot(ts, s["GPU Core Temperature (Celsius Degree)"], "-", color=col, lw=1.4)

a.set_ylabel("Throughput (TFLOP/s)"); a.set_ylim(140, 230)
b.set_ylabel("GPU frequency (MHz)"); b.set_ylim(600, 1300)
c.set_ylabel("Stack power (W)"); c.set_ylim(150, 300)
c.axhline(215, color="0.4", lw=0.8, ls="--")
d.set_ylabel("Temperature (°C)"); d.set_ylim(30, 70)
for ax in axs.flat:
    ax.set_xscale("log"); ax.set_xlim(0.01, 20)
for ax in (c, d):
    ax.set_xlabel("Time since first call (s)")
for ax, l in zip(axs.flat, "abcd"):
    ax.text(0.02, 0.96, f"({l})", transform=ax.transAxes, va="top", fontsize=12)
fig.legend(*a.get_legend_handles_labels(), loc="upper center", ncol=1, fontsize=10.5, frameon=False,
           bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(f"{OUT}/power_phases.png", dpi=300)
print("ok")
