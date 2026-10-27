#!/usr/bin/env python3
"""Write the E9 points not yet measured in runs/e9_c*/ and the payload for chunk N. Usage: python3 analysis/e9_plan.py <N>"""
import sys, os, glob, csv
QA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = int(sys.argv[1])
kgrid = [(2048, 2048, k) for k in list(range(64, 16385)) + list(range(16385, 131072, 15))]
mkgrid = [(2 ** e, 4096, 2 ** e) for e in range(5, 18)]
done = set()
for f in glob.glob(f"{QA}/runs/e9_c*/e9*_full.csv"):
    for r in csv.DictReader(open(f)):
        if r["phase"] == "point":
            done.add((int(r["m"]), int(r["n"]), int(r["k"])))
krem = [p for p in kgrid if p not in done]; mkrem = [p for p in mkgrid if p not in done]
kf, mf = f"{QA}/e9_plan/chunk{N}_ksweep.txt", f"{QA}/e9_plan/chunk{N}_mk.txt"
os.makedirs(f"{QA}/e9_plan", exist_ok=True)
for fn, pts in [(kf, krem), (mf, mkrem)]:
    assert not os.path.exists(fn), f"{fn} exists (plans are never overwritten)"
    with open(fn, "w") as f:
        f.write("".join(f"{m} {n} {k}\n" for m, n, k in pts))
tmpl = open(f"{QA}/experiments/e9_chunk_template.sh").read()
pay = f"{QA}/experiments/e9_chunk{N}.sh"
assert not os.path.exists(pay)
open(pay, "w").write(tmpl.replace("@KFILE@", f"$QA/e9_plan/chunk{N}_ksweep.txt").replace("@MFILE@", f"$QA/e9_plan/chunk{N}_mk.txt").replace("@N@", str(N)))
print(f"chunk {N}: {len(krem)} K-sweep points remaining (first {krem[0] if krem else None}), {len(mkrem)} MK points; payload {pay}")
