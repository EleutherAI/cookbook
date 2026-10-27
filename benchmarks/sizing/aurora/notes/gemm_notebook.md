# Aurora GEMM experiments: lab notebook

*Experiment-by-experiment record of the October 2026 runs (E0-E11, attention in `attention.md`). Each section gives the question,
the prediction, what was measured, and a verdict. Run directories (`runs/<name>_<date>/`) are the raw outputs on Aurora; the processed
CSVs and summaries are in `../data/<experiment>/`, the scripts in `../bench/`, `../attn/`, `../experiments/` and `../analysis/`.
"Original sweep" means the September 2026 sizing sweeps that produced the first draft's figures.*

*Script names are as they were run. Since then `bench/freq_log.sh` became `bench/smi_log.sh`, the generated
`experiments/e9_chunk{1..5}.sh` and the E11a rerun plan were left out, and `analysis/plot_e9.py` became `analysis/merge_e9.py`.*

## Round 2 summary (2026-10-26: E9, E10, E11a, E8; details at the end of this file)
1. **E9 — new steady-state Figs. 2/3 done** (23,967 + 13 points, `e9/`): aligned k sustain ~158-163 TFLOP/s from k ~ 4k to 131k on one stack; the old
   ~200 plateau was boost (old data 8-21% high for 4k <= k < 41k, exact for k < 1k and k >= 41k). k-slicing replay agreement 100% over 1,502
   streaming points. Fig. 3 m=k >= 8192 at 169 (original sweep within 2%); m=k = 2048/4096 are too short to leave the boost even in this protocol.
2. **E10 — the 215 W limit is per stack, plus a per-GPU limit**: one stack per GPU holds 215.7 W each with no cross-GPU effect; both stacks of
   a GPU busy -> ~404-411 W per GPU (~200 W per tile, "Current Excursion"), costing 1-6% (k=16128) and ~5% median / 14% worst (k=98562).
3. **E11a — ~24% of the original transformer GEMM medians sit in the boost window** (all 1,151 points > 185 TFLOP/s are < 0.5 s runs); those runs also
   used 12 concurrent stacks. Proposed rerun: 224 shapes, ~27 min on one stack (~0.45 node-h), shape files in `e11a/rerun_plan/` — not run.
4. **E8 — cache-line splitting confirmed**: same kernel and instruction counts at 98560 vs 98564, but +58% HBM reads, +93% L3 fills, 60% more
   cycles, load/store queue saturated; the power cap raises the clock of the stalled kernel, so the time penalty is 27%.

| Exp | Verdict | Run directory |
|---|---|---|
| E9 | delivered (CSV in the sizing-benchmark format + full stats + PNGs) | `runs/e9_c{1..5}_*`, merged `e9/` |
| E10 | cap per stack (215 W) + per-GPU limit (~405 W) when both stacks busy | `runs/e10_20261026_053256/` |
| E11a | 23-24% of points boost-contaminated; rerun proposed (not run) | `e11a/` |
| E8 | cache-line-splitting hypothesis confirmed | `runs/e8_20261026_054558/` |

## Summary (round 1)
1. **Kernel selection replay confirmed**: oneDNN v3.10.2; odd k -> copy-based `jit:xe_hp`, even k -> `jit:gemm:any`; k=98562 uses catalog entry 463 (verbose string
   identical to the replay); persistent 56-WG grid; k-slicing visible as +5 kernel args and a 14.7 MB memset (E1, E2). Copy/no-copy matched 126/126 transformer shapes (E6).
2. **k = 2 mod 4 square wave confirmed**: low 106-109 / high 124-125 TFLOP/s exactly when ceil(k/256) mod 7 in {1,2,3}; replay agreement 99.78% (median-based 100%);
   runs of 768/1024, transitions at k mod 1792 = 2 / 770; holds on all 12 stacks (E3, E5). Alignment-class levels within ~2% of the original CSV (incl. odd ~100.5).
3. **"Two regimes" = power-limit boost, not L2 (L2 hypothesis refuted)**: each stack runs at ~1.0-1.2 GHz / ~270 W for ~0.3 s after idle, then is held at ~215 W,
   800-850 MHz, 157-172 TFLOP/s, with or without L2 reuse (rotation costs 1-4%). The original protocol (short 250-call runs after a CPU-init idle) reproduces its
   ~200 -> ~160 step at k ~ 24k-41k within ~2% (E4, E4b, E4c). Also: its CSVs are median over all calls incl. warm-up, not amin (E0).
4. **Variance larger than assumed**: 4-6% stack-to-stack within one node (9% for 0 mod 32), consistent ranking across classes; similar across nodes (E5) — the 1.8% prior is refuted.
5. **Transformer predictions mixed**: AOV BMM sawtooth (period 256 in dh, peaks 256/512) confirmed (E7, Spearman 0.89); vocab 50,257 dgrad -32% confirmed, but wgrad is
   copy-based at 50,304 too (only -2.5%); forward h_out ranking not predicted by the replay's wave x padding efficiency (Spearman 0.24) although 1792-multiples win by 1-5% (E6, E6b).

| Exp | Verdict | Run directory |
|---|---|---|
| E0 | answered (Aurora = ed7e278 + uncommitted min->median; FLAT confirmed) | `runs/e0_login/`, `runs/e012_20261025_211815/` |
| E1 | confirmed | `runs/e012_20261025_211815/` |
| E2 | confirmed | `runs/e012_20261025_211815/` |
| E3 (+E3b odd) | confirmed | `runs/e3_20261025_212659/`, `runs/e47_20261025_220432/e3b_*` |
| E4 (+E4b, E4c) | L2 hypothesis refuted; clock mapping confirmed; cause = power-limit boost window | `runs/e47_20261025_220432/`, `runs/e4b_20261025_221835/`, `runs/e4c_20261025_222747/` |
| E5 | refuted (spread 4-6%, not ~1.8%) | `runs/e5_20261025_223736/` |
| E6 (+E6b) | paths confirmed; vocab partly; h_out ranking refuted | `runs/e6_20261025_231434/`, `runs/e6b_20261025_232217/` |
| E7 | confirmed | `runs/e47_20261025_220432/` |

**Payload changes during round 1** (originals kept on Aurora): `bench/freq_log.sh` (tile 0; `--ims` needs `--file`, which needs a pty -> E4c uses `script`);
E4 1000 calls/shape; E6 + one-call ONEDNN_VERBOSE pass; added payloads `e3b_odd.sh`, `e4b_clock.sh`, `e4c_clock.sh`, `e6b_sustained.sh`.

## E0 — Provenance and environment (login part: `runs/e0_login/`)
- **Aurora cookbook copy**: HEAD = `ed7e2782` ("add profiling options and support for xpu to sizing benchmarks", 2026-08-12),
  branch `profile_and_xpu` = the pushed commit `ed7e278`, **plus uncommitted local edits** to `utils.py`,
  `megatron_wrapper.py`, `megatron/model/transformer.py` (`git_diff.txt`).
- **benchmark_mm statistic**: local edit changed `np.amin(times)` -> `np.median(times)` in every `benchmark_*`
  (the change is **min->median**). `benchmark_mm` still takes the statistic over **all**
  warm-up + measured iterations (no `times = times[num_warmup:]` line; the other benchmarks do drop warm-up).
  A/B are still created with CPU `torch.randn(...).bfloat16().to(device)`. So which statistic produced the CSVs in
  `median/` depends on when they were generated relative to this uncommitted edit. `utils.py` mtime is
  2026-09-17 01:38 UTC and `median/basicGemmKSweep.csv` was started 2026-09-20 18:43 (header: torch 2.10.0a0+git449b176,
  frameworks 2025.3.1, node x4418c6s1b0n0), so **the K-sweep CSV is almost certainly median over all 250 calls incl.
  warm-up, not amin** as the prior analysis assumed. (Median vs min matters for the bimodal/k-slicing classes.)
- **setup.sh**: `module load oneapi/release/2025.3.1 mpich/opt/5.0.0.aurora_test.3c70a61 frameworks/2025.3.1`
  (modulefile under `/opt/aurora/26.26.0`), sets `CPU_BIND` and prepends a DeepSpeed checkout to `PYTHONPATH`.
  No `ONEDNN_*` variables. The frameworks module sets `ZE_FLAT_DEVICE_HIERARCHY=FLAT` (login default is COMPOSITE),
  `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1`, `ZE_ENABLE_API_TRACING=1`, `ONEAPI_DEVICE_SELECTOR=opencl:gpu;level_zero:gpu`.
  Python = frameworks 2025.3.1, **torch 2.10.0a0+git449b176**, torch.version.xpu 20250302.
- `xpu-smi dump` on this version has no `--ims`; `freq_log.sh` will fall back to `-i 1` (1 s sampling).
- `unitrace` is on PATH after setup.sh (pti-gpu 0.17.0).
- **Compute part** (`runs/e012_20261025_211815/`, job 8965299, node x4105c6s0b0n0, 2 min): `ZE_FLAT_DEVICE_HIERARCHY=FLAT`,
  **12 XPU devices visible => FLAT mode** confirmed; each device = Max 1550 stack, 448 EUs, 56 subslices, 65,520 MB;
  driver 1.6.33578+77 (the September sweep header: +42, a different driver build).
- Verdict E0: Aurora copy = pushed `ed7e278` + uncommitted min->median edit; FLAT confirmed.

## E1 — oneDNN version and implementation per alignment class (`runs/e012_20261025_211815/`)
- **Question**: which oneDNN and which implementation runs per k class? **Prediction**: v3.10.x; odd k -> `jit:xe_hp:gemm:any`; even k -> `jit:gemm:any`.
- **Measured**: `oneDNN v3.10.2 (commit f1d47193)`. k=98561 (odd) -> `jit:xe_hp:gemm:any`; k = 98560, 98562, 98564, 99330, 41218, 41986 -> `jit:gemm:any`.
  `ONEDNN_VERBOSE=all` at k=98562 shows xe_hp being skipped (`jit_xe_hp_systolic.cpp:85`) and prints the
  **catalog entry string `aB16 aB16 aB wg 4x8 cab4x2 ks16 xaf vav hi pt sr br bk0 sn dm grf256 sys rr kv afb`, identical to the
  replay's entry 463 strategy** (kernel line: `... wg 4x8 sys xaf k64 grf256 ... kv afb ...`).
  `debuginfo=5`: 0 `consider:` lines (release build, as expected).
- **Verdict: confirmed** (version, both paths, and entry 463 for k=98562).

## E2 — Kernel names, counts, launch geometry (`runs/e012_20261025_211815/`, unitrace `-d -v`)
- **Prediction**: odd k adds copy/pack kernels; main kernels persistent (~56 WGs); 98562 vs 99330 differ visibly.
- **Measured** (per call):
  | k | kernels | main grid {global WGs}{local} | main kernel args / SLM | extra |
  |---|---|---|---|---|
  | 98560 (0 mod 32) | `gemm_kernel` | {56;1;1} {128;4;1} | 19 args, SLM 0 | — |
  | 98561 (odd) | `xe_hpc_systolic_gemm_copy` {97;128;1} (2.96 ms) + `xe_hpc_systolic_gemm_copy` {8;64;1} (18 us) + `gemm_kernel` {56;1;1} (3.96 ms) | {56;1;1} {128;4;1} | 19 args | 2 copy/pack kernels = 41% of device time |
  | 98562 (pred. k-slicing ON) | `gemm_kernel` | {56;1;1} {128;4;1} | **24 args**, SLM 64 KiB | **`zeCommandListAppendMemoryFill` of 14,680,064 B** (= 2048 x 1792 x fp32, zeroing for the atomic fix-up) |
  | 99330 (pred. k-slicing OFF) | `gemm_kernel` | {56;1;1} {128;4;1} | 19 args, SLM 64 KiB | — |
  Local size 128x4 = 512 work-items = 32 SIMD16 subgroups = the 4x8 subgroup layout; 56 WGs = one per Xe-core (persistent).
- **Verdict: confirmed** on all three points. The k-slicing difference is visible as 5 extra kernel arguments plus a
  14.7 MB memory fill before each call at 98562 but not 99330.
- Throughput note (few calls, not the E3 statistic): k=98560 ran at 218-242 TFLOP/s, 98562 at 117-147, 99330 at 134-174,
  98561 at 117-120. The aligned case is far above the prior streaming level (~161); E3 checks this properly.

## E3 — One k-slicing period with corrected statistics (`runs/e3_20261025_212659/`, job 8965318, 31 min)
- **Question**: does the k = 2 mod 4 square wave follow ceil(k/256) mod 7 in {1,2,3} exactly, at the predicted levels?
  **Prediction**: low ~108 / high ~125; replay agreement >= 99%; 32|k ~161; odd ~102; old-style within ~2% of the original CSV.
- **Measured** (k = 98304..100096 step 2, 200 calls after 50 warm-up, device init, stack 0; values = median over k of the per-k statistic):
  | class | n | TFLOP/s (median time) | TFLOP/s (min time) | prior (original sweep, streaming) |
  |---|---|---|---|---|
  | 0 mod 32 | 57 | 157.8 | 164.3 (max 224.5) | 160.7 |
  | 16 mod 32 | 56 | 139.4 | 146.6 | 142.3 |
  | 8 mod 16 | 112 | 130.1 | 138.1 | 132.8 |
  | 4 mod 8 | 224 | 126.2 | 134.5 | 128.6 |
  | 2 mod 4, r in {1,2,3} | 192 | 106.1 | 109.4 | 108.3 |
  | 2 mod 4, r in {0,4,5,6} | 256 | 123.8 | 125.2 | 125.0 |
  (r = ceil(k/256) mod 7.) `compare_replay.py`: **agreement 99.78% of 448 points** (min-based; the only miss, k=98598, has a
  fast min but median 106.1, i.e. low level); **median-based agreement 100%**. Transitions at k = 98562 (k mod 1792 = 2, down)
  and 99330 (mod 1792 = 770, up): low run 98562..99328 = **768 long**, high runs 1024 long — exactly as predicted.
  Entry 443 (2D block) with vs without k-slicing: 136.0 vs 136.0 (min-based medians) — the mode costs ~0% there (prior: ~1%).
- **Old-style replicate** (host init, step 14, 129 points) vs the original CSV at the same k: median |dev| 1.29%, p90 1.94%,
  93% of points within 2%; systematic offset -1.2..-1.6% in every class (new node/driver slightly slower). Device-init vs
  host-init medians differ by 0.06% -> the CPU-init idle gap does not matter at this size.
- **Odd k**: not covered by E3's step-2 grid (all even). Spot checks at k=98561 in E1/E2 (3 runs of 5 calls) gave
  **117-120 TFLOP/s vs the original 102.5** at the same k (unitrace: copy kernel 2.96 ms + gemm 3.96 ms = 6.9 ms/call).
  **Rechecked in E3b** (`runs/e47_20261025_220432/e3b_odd*.csv`, k = 98305..100099 step 14, 129 points, 200 calls):
  **odd median 100.5 TFLOP/s (min-based 102.8) vs the original 102.2 at the same k (-1.7%)**; host-init subset (19 points) identical.
  The 117-120 spot values were the post-idle boost transient (see E4): 5-call runs sit inside the ~10-40 ms fast window.
- **Note on min vs median**: per-k min is noisy for aligned k (0 mod 32 min up to 224.5 while median 157.8): occasional
  much faster calls exist even in the streaming regime. Relevant to E4 (two regimes).
- **Verdict: confirmed** (square wave, levels, period, run lengths, transition points, replay >= 99%). Aligned-class levels
  confirmed within ~2%. Odd ~102: **confirmed** after the E3b recheck (100.5, -1.7%).

## E4 — L2 residency vs kernel duration (first run: `runs/e47_20261025_220432/`, job 8965390, node x4611c7s0b0n0)
- **Question**: is the ~200 vs ~161 TFLOP/s split (k < ~24k vs k > ~41k) caused by L2 residency or by sustained-load clock?
  **Prediction (L2)**: `rotate800` drops the L2-resident plateau ~200 -> ~160 and the clock falls; `burst20` stays ~200.
  **Prediction (duration)**: `burst20` falls to ~160 without rotation; `rotate800` changes little.
- **Clock logging failed in this run**: this xpu-smi accepts `--ims` only together with `--file` ("--ims requires --file").
  `bench/freq_log.sh` fixed (tile-specific `-t 0`, `--ims 100 --file`, + metric 35 throttle reason) and E4 re-run as E4b (below).
  Changes vs the first payload: `-t 0` (FLAT stack 0; device-level frequency averages both tiles), 1000 calls / 100 warm-up
  per shape (so each shape spans >= 1 xpu-smi sample). Originals saved in `runs/e0_login/*.orig`.
- **Measured TFLOP/s (median of measured calls)**, m = n = 2048:
  | k | footprint MB | reuse | rotate800 | burst20 |
  |---|---|---|---|---|
  | 5376 (=3x1792) | 52 | 202.8 | 189.8 | 187.8 |
  | 8960 | 82 | 167.7 | 160.8 | 158.7 |
  | 12544 | 111 | 170.8 | 158.1 | 160.6 |
  | 16128 | 141 | 164.0 | 157.2 | 160.0 |
  | 19712 | 170 | 164.6 | 160.9 | 160.5 |
  | 23296 | 199 | 165.0 | 160.0 | 159.3 |
  | 50176 | 419 | 156.8 | 156.1 | 154.4 |
  (+32 neighbours, 32 mod 64 class, run 3-10% lower in the same pattern.) `vary_m` (k = 16128): m = 512 181.5, 1024 165.4, 2048 162.3, 4096 159.8.
- **Per-call timeline (`e4.csv.iters.csv`)**: after every idle gap (operand allocation between shapes) the first ~15-35 calls
  run at **200-250 TFLOP/s** for ~10-40 ms of device time, then drop (sometimes to ~100-140 for 1-3 calls) to a flat
  155-170 that does not change for the remaining 0.5-3 s. E.g. reuse k=16160: calls 1-27 at 171-238, then 99, 99, then 157.5.
  The same transient appears with rotate800 (L2 defeated) and in vary_m (m=4096, 214 TFLOP/s for 40 ms then 159.8).
- **Interim verdict**: in steady state there is **no ~200 L2-resident plateau**: L2-resident sizes sustain only 164-171 under
  reuse; defeating L2 costs 2-7%, not ~20%. burst20 sits at ~160 with no rotation. -> **L2 hypothesis refuted; the
  duration/transient picture is supported**: ~200+ only within tens of ms after idle. The original 250-call medians for small k
  (whole run ~0.1-0.3 s, preceded by a CPU-init idle) would contain a large share of boost-phase calls; larger k would not.
  Clock evidence pending (E4b).

## E7 — Attention-over-values BMM, dh sweep (`runs/e47_20261025_220432/e7*.csv`)
- **Question/Prediction**: for dh > 128 throughput is a sawtooth with period 256 in dh, peaks at dh = 256 and 512 (tile height 256);
  dh <= 128 uses a 128-row tile.
- **Measured** (median TFLOP/s; out[b, s, dh] = P[b, s, s] @ V[b, s, dh], dh = 32..512 step 16):
  | config | dh=128 | 144 | 256 | 272 | 512 | Spearman(meas, replay eff), dh>128 |
  |---|---|---|---|---|---|---|
  | b=16, s=2048 | 114.2 | 133.0 | **165.3** | 103.6 | **166.1** | 0.894 |
  | b=32, s=4096 | 75.9 | 81.8 | **132.3** | 97.0 | **149.9** | 0.891 |
  Peaks at 256 and 512, cliff at 272 (just past a multiple of 256), monotone-ish rise in between: **period-256 sawtooth confirmed**.
  dh <= 128: replay picks 128x128 (entry 449) for b16/s2048 dh 48-128 and for b32/s4096 dh 80-128; smaller tiles (64x64, 32x32) at dh <= 64 for b32/s4096.
  Secondary effect not in the model: dh multiples of 32 beat their +16 neighbours above 256 (e.g. 352 > 368, 416 > 432, 480 > 496), i.e. row alignment of V/out (2*dh bytes).
  Tile identities are replay predictions; this run did not log ONEDNN_VERBOSE kernel names.
- **Verdict: confirmed** (sawtooth period 256 with peaks at 256/512; 128-row tile for most dh <= 128, smaller tiles at dh <= 64 for the larger problem).

### E4b — repeat + idle sweep + old-protocol reconstruction (`runs/e4b_20261025_221835/`, job 8965419, node x4404c5s2b0n0)
- Clock log failed again: `--file` mode reads ESC from a TTY and exits under PBS ("tcgetattr(): Inappropriate ioctl for device").
  Fixed with a pty wrapper in E4c.
- **Repeat on another node**: same pattern, levels ~4-6% higher than E4's node (reuse 171-175 for L2-resident 0 mod 256 k,
  rotate800 165-169, burst20 168-170, k=50176 163-166; k=5376 216.6/196.9/193.9). vary_m: 173, 167, 172, 170.
  -> Again no ~200 steady-state plateau; rotating operands out of L2 costs 1-4%.
- **Idle sweep** (k = 16128, 2000 back-to-back calls per shape): first 20 calls 231-238 TFLOP/s; fast phase (>185 TFLOP/s) lasts
  **~225 ms of device time after a 0.2 s idle and ~350-400 ms after >= 1 s idle** (3 s and 10 s idle give no more); afterwards
  171.6-171.7 in every case. On E4's node the fast window was only ~10-40 ms -> **node-to-node difference in the boost budget**.
- **Sustained 20,000-call run** (k = 16128, ~17 s): 0-50 ms 238.6; 50-500 ms 194.3; 0.5-2 s 171.3; 2-5 s, 5-10 s, 10-20 s 171.3.
- **Old protocol reconstructed** (host init, statistic = median over all 250 calls incl. warm-up, as in the uncommitted utils.py edit):
  | k | 4096 | 8192 | 12288 | 16384 | 20480 | 24576 | 28672 | 32768 | 36864 | 40960 | 49152 | 65536 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | this run (median of 250) | 196.5 | 205.0 | 205.9 | 196.2 | 196.3 | 189.3 | 185.3 | 173.4 | 167.2 | 163.2 | 163.4 | 162.5 |
  | original CSV | 211.6 | 201.5 | 206.7 | 197.0 | 195.5 | 199.4 | 197.1 | 184.6 | 177.8 | 159.2 | 158.9 | 156.0 |
  | 250-call run length (ms) | 61 | 88 | 150 | 178 | 226 | 284 | 336 | 396 | 444 | 498 | 670 | 810 |
  The two regimes **reappear with the old protocol**, matching the original sweep within ~2% at k <= 20k and above 40k; the transition
  sits where the 250-call run (~0.25-0.5 s) outlasts the post-idle fast window, so the median flips from fast to sustained calls
  (the original node held the boost slightly longer: its 24-37k points stay high a bit further).
- **Verdict (E4 so far)**: **L2-residency hypothesis refuted**. The ~200 regime is a **time-limited post-idle boost** (a few hundred ms
  of high throughput, refilled by ~1 s idle), and the old K sweep shows it because each point is a short run after a CPU-init idle.
  Sustained bf16 GEMM throughput on one stack is ~157-172 TFLOP/s for aligned shapes regardless of L2 residency.

### E4c — clock and power (`runs/e4c_20261025_222747/`, job 8965434, node x4404c5s2b0n0; `e4c_summary.txt`)
- xpu-smi tile 0 at 100 ms (`--ims 100 --file` under a pty via `script`), aligned to per-call host timestamps.
  Idle: 1500 MHz (reported), 86 W, 37 C, not throttled.
  | run (k) | window after start | TFLOP/s | GPU MHz | tile power W | throttle reason |
  |---|---|---|---|---|---|
  | reuse 16128 (L2-resident) | 0-0.3 s | 202-205 | 1000-1075 | ~270 | Average Power + Current Excursion |
  | | 0.6-17 s | 171.4 | **850** | **216** | Average Power Excursion |
  | rotate800 16128 (L2 defeated) | 0-0.3 s | 193-196 | 950-1200 | 98-268 | -> Power/Current Excursion |
  | | 0.6-17 s | 167.0 | **850** | **215** | Average Power Excursion |
  | reuse 50176 (streaming) | 0-0.3 s | 190-217 | 1100-1225 | ~268 | Power/Current Excursion |
  | | 0.6-21 s | 167.3 | **800** | **216** | Average Power Excursion |
  | old protocol k=8192 (250 calls, ~90 ms) | whole run | 204-205 | 1100-1200 | ~271 | Current/Avg Power Excursion |
  | old protocol k=49152 (~0.67 s) | 0-0.3 s / 0.3-0.67 s | 188-221 / 163-166 | 800-1500 / 800-1100 | 270-300 / 215-261 | Avg Power Excursion |
  Steady-state throughput per clock = 197-209 TFLOP/s per GHz = 0.86-0.91 of the 229 TFLOP/s/GHz XMX peak (the prior analysis assumed ~0.87).
- **E4 verdict**: the prior's numeric mapping is **confirmed** (~200 TFLOP/s <-> ~1.0-1.2 GHz, ~160-170 <-> 0.80-0.85 GHz), but the cause is
  **not L2 residency (refuted: same 850 MHz and only -2.5% TFLOP/s with L2 defeated)**. It is the **power limit**: each stack may draw ~270 W
  for ~0.3 s after idle (power/current excursion), then is held at ~215 W average, which pins the clock at 800-850 MHz. The old
  K sweep's "L2-resident ~200" regime is an artefact of short (< ~0.3 s) measurement runs after an idle gap; candidate (2)
  "size-dependent sustained clock" is also not it — it is time-dependent, set by run length, not by m*n*k.
  Recommendation for the paper: report sustained (>= 1 s back-to-back) numbers, or state the boost window explicitly.

## E5 — Stack-to-stack variance (`runs/e5_20261025_223736/`, job 8965464, node x4306c7s6b0n0, 31 min; `e5_summary.txt`)
- **Question**: how much do the 12 stacks of one node differ? **Prior evidence**: two cross-node overlap points agreed within 1.8%.
- **Measured** (k = 41216..43008 step 2, 50 calls after 10 warm-up per k, stacks run one after another; per-stack median over k of median-time TFLOP/s):
  | class | min stack | max stack | median | (max-min)/median |
  |---|---|---|---|---|
  | 0 mod 32 | 156.9 (s4) | 172.0 (s3) | 162.8 | 9.3% |
  | 16 mod 32 | 142.1 (s4) | 148.5 (s3) | 145.4 | 4.4% |
  | 8 mod 16 | 132.6 (s4) | 138.1 (s3) | 135.8 | 4.1% |
  | 4 mod 8 | 127.9 (s4) | 133.7 (s3) | 131.5 | 4.4% |
  | 2 mod 4, high | 123.6 (s4) | 130.8 (s3) | 127.6 | 5.7% |
  | 2 mod 4, low (k-slicing) | 107.1 (s4) | 111.4 (s3) | 109.9 | 3.9% |
  Ordering is consistent across classes (stack 4 lowest, stack 3 highest in every class) -> a per-stack property (most likely the
  sustained power-limited clock, cf. E4c), not noise. Min-time based spreads: 4.2-5.9%.
  The k-slicing square wave holds on every stack (agreement 99.1-99.8%, low/high 107-111 / 124-131).
- Cross-node: the same shapes in E4 (x4611c7s0b0n0) and E4b (x4404c5s2b0n0) differed by 4-6% as well.
- **Verdict: refuted** (the ~1.8% agreement does not generalise): expect **~4-6% stack-to-stack spread (up to ~9% for the best-aligned class)**
  within one node, and similar across nodes. Single-stack curves should carry a ±3-5% band, or report the median of several stacks.

## E6 — Transformer-shaped GEMMs vs predictions (`runs/e6_20261025_231434/`, job 8965554, node x4415c3s3b0n0; `e6_summary.txt`, `e6.joined_impl.csv`)
- **Question/Prediction**: forward NT with h_in = 4096: h_out in {3584, 5376, 7168} at 4,096-8,192 tokens beat power-of-two neighbours by the
  predicted wave x padding efficiency; vocab 50,257: forward stays no-copy, both backward GEMMs `jit:xe_hp:gemm:any` and lose ~1/3 vs 50,304.
  Report Spearman(measured, predicted) per layout.
- **Implementation per shape** (separate 1-call `ONEDNN_VERBOSE` pass, added to the payload): **126/126 shapes match the replay's
  copy-based/no-copy prediction**. All 120 grid shapes -> `jit:gemm:any`.
- **Vocab** (8192 tokens x 4096 hidden; 20 calls after 5 warm-up):
  | GEMM | 50,257 impl | 50,257 TFLOP/s | 50,304 impl | 50,304 TFLOP/s | 50,257 vs 50,304 |
  |---|---|---|---|---|---|
  | fwd logits (NT) | jit:gemm:any | 158.8 | jit:gemm:any | 179.3 | -11% |
  | dgrad (NN) | **jit:xe_hp:gemm:any** | 119.7 | jit:gemm:any | 179.3 | **-33%** |
  | wgrad (TN) | **jit:xe_hp:gemm:any** | 153.1 | **jit:xe_hp:gemm:any** | 159.6 | -4% |
  Paths **confirmed** for 50,257 (fwd no-copy, both backward copy-based). The ~1/3 loss is **confirmed for dgrad only**: wgrad at 50,304
  is *also* copy-based (the replay predicts this: large-TN rule, m*n >= 12800^2), so padding to 50,304 does not help wgrad (-4%).
  Fwd 50,257 is 11% slower despite staying no-copy (misaligned C only). Caveat: these 25-call runs last ~0.5 s and may partly sit in the boost window.
- **Grid** (median TFLOP/s; forward rows = tokens, columns = h_out):
  | tokens \ h_out | 2048 | 3072 | 3584 | 4096 | 5376 | 7168 | 8192 | 12288 | 14336 | 16384 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 4096 | 194 | 188 | 191 | 162 | 168 | 169 | 161 | 163 | 162 | 162 |
  | 8192 | 164 | 166 | 169 | 161 | 163 | 166 | 165 | 161 | 161 | 161 |
  | 14336 | 168 | 164 | 163 | 165 | 162 | 162 | 162 | 161 | 161 | 161 |
  | 16384 | 160 | 164 | 166 | 164 | 162 | 162 | 161 | 161 | 161 | 161 |
  Predicted efficiencies span only 0.914-1.000 (wgrad: all 1.000). Spearman(measured, predicted eff): fwd 0.25, dgrad -0.49,
  wgrad undefined (constant prediction). Spearman(measured, shape run length) = -0.46 to -0.67: the fastest cells are the *first,
  shortest* shapes of each process (T = 4096, h_out <= 3584 at 188-199), i.e. the post-idle power boost from E4, not tiling.
  Directionally, 3584 > 4096 and 5376/7168 > 8192 at 4,096-8,192 tokens (+2-5%; predicted +1.5%), but this is within the boost
  confound and the 4-6% stack spread.
- **Verdict**: kernel-path predictions **confirmed** (126/126); vocab: **partly confirmed** (dgrad -33% confirmed; wgrad not, because 50,304
  is copy-based as well); forward ranking **inconclusive** in this run (confounded by the boost) -> re-measured in steady state as E6b.

### E6b — same grids in steady state (`runs/e6b_20261025_232217/`, job 8965575, node x4308c2s5b0n0; `e6b_summary.txt`)
- Each grid preceded by a ~1.3 s heater shape (16384x4096x16384), no idle between shapes, then repeated in reverse order.
  **Reproducibility forward vs reverse pass: median |diff| 0.09-0.15%** (max 1.7-2.2%, one 16.6% outlier in wgrad) -> on one stack in
  steady state, differences of a few % are real. The E6 boost cells are gone (T=4096, h_out=2048: 194 -> 163).
- **Forward (NT)**, steady-state median TFLOP/s:
  | tokens \ h_out | 2048 | 3072 | 3584 | 4096 | 5376 | 7168 | 8192 | 12288 | 14336 | 16384 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 4096 | 163.2 | 161.3 | 166.1 | 162.2 | 168.4 | 168.6 | 161.1 | 163.5 | 162.5 | 162.2 |
  | 8192 | 162.3 | 166.1 | 169.3 | 161.3 | 163.3 | 163.8 | 163.7 | 161.2 | 161.7 | 161.1 |
  | 14336 | 168.0 | 164.7 | 162.8 | 165.0 | 163.1 | 161.8 | 161.6 | 161.2 | 161.6 | 161.4 |
  | 16384 | 160.4 | 163.1 | 162.5 | 162.1 | 161.0 | 161.6 | 160.9 | 160.5 | 160.5 | 160.3 |
  Multiples of 1792 vs power-of-two neighbours, measured (predicted): T=4096: 3584 vs 4096 +2.4% (+0.0), 5376 vs 4096 +3.9% (+0.0),
  7168 vs 8192 +4.7% (+1.6); T=8192: 3584 vs 4096 +5.0% (+1.6), 5376 vs 4096 +1.2% (+0.0), 7168 vs 8192 +0.0% (+1.0).
  The largest predicted gap (T=4096, h_out 2048 predicted 9.4% worse than 3584) is only 1.8%.
  14,336 vs 16,384 tokens at power-of-two widths: +0.5..+1.8% (predicted +1.0..+1.1%).
  **Spearman(measured, predicted eff): fwd 0.24, dgrad -0.46 (n = 40 each); wgrad prediction constant (all 1.000).**
- **Vocab, steady state**: dgrad 50,257 **110.0** vs 50,304 **161.2 (-32%)**; fwd 143.6 vs 161.6 (-11%); wgrad 149.3 vs 153.2 (-2.5%, both copy-based).
- **E6 verdict (final)**: kernel paths **confirmed** (126/126). Vocab: dgrad loses ~1/3 **confirmed**; "both backward GEMMs lose ~1/3" **refuted
  for wgrad** (50,304 wgrad is copy-based too, so padding gains only 2.5%); fwd -11% not predicted. Forward h_out ranking: **refuted as a ranking**
  (Spearman 0.24; dgrad negative) — multiples of 1792 do tend to beat power-of-two neighbours (5 of 6 comparisons, +1..+5%), but the
  replay's wave x padding efficiency does not predict the magnitudes; the remaining 2-5% structure comes from something the model omits
  (likely the same alignment/cache effects as the 8->64 B gain in the prior analysis).

---
# Round 2 (2026-10-26): E9, E10, E11a, E8

## E9 — Steady-state data for Figs. 2 and 3 (`runs/e9_c{1..5}_*`, merged products in `e9/`)
- **Protocol**: stack 0, `torch.mm`, bf16, device init; operands are views of flat device buffers allocated once (row stride = k, so each
  point keeps its alignment class), so nothing is allocated between points; **>= 2.0 s pre-heat** (k = 16128 back to back) at the start of
  every process; anchors k = 8192, 16384, 32768, 65536, 98560-98563 right after the pre-heat and again at the end of every chunk; 50 warm-up +
  200 measured calls per point (event pair + sync per call, as before); min/p10/median/p90/max recorded. Script `bench/steady_sweep.py`,
  chunk planner `analysis/e9_plan.py` (each chunk resumes from the points not yet measured), payloads `experiments/e9_chunk{1..5}.sh`.
- **Grids**: Fig. 2 k = 64..16384 step 1 + 16385..131060 step 15 (23,967 points, all done); Fig. 3 m = k = 2^5..2^17, n = 4096 (13 points, chunk 5).
- **Outputs** (`e9/`): `e9_fig2_basicGemmKSweep.csv`, `e9_fig3_basicGemmMKSweep.csv` (the sizing-benchmark `m,n,k,throughput` format, throughput = TFLOP/s at the
  median time), `e9_fig2_full.csv(.gz)` / `e9_fig3_full.csv` (all statistics + node + chunk), `e9_anchors.csv`, `e9_summary.txt`;
  figures `fig2a_k_sweep_small.png` (k < 2^10), `fig2b_k_sweep_medium.png` (k < 2^14), `fig2c_k_sweep_large.png` (all k), `fig3_mk_sweep.png`
  (+ `*_vs_orig.png` with the old data as dashed same-colour EMAs). Style from the sizing figures notebook (Dark2, colour = largest power
  of 2 dividing k capped at 64 as in the notebook, scatter + EMA(20), font 18, dpi 500) with dotted lines at multiples of 1792.
- **Chunk comparability (anchors, median TFLOP/s at chunk start)**:
  | chunk | node | k=8192 | 16384 | 32768 | 65536 | 98560 | 98561 | 98562 | 98563 | k covered |
  |---|---|---|---|---|---|---|---|---|---|---|
  | c1 | x4412c3s1b0n0 | 162.3 | 159.6 | 148.4 | 149.8 | 152.2 | 97.9 | 103.3 | 98.1 | 64-16910 |
  | c2 | x4517c7s3b0n0 | 163.5 | 160.4 | 157.8 | 159.9 | 160.9 | 102.3 | 108.0 | 102.3 | 16925-71345 |
  | c3 | x4113c4s4b0n0 | 171.2 | 169.1 | 157.8 | 159.6 | 160.4 | 102.7 | 107.4 | 102.1 | 71360-99680 |
  | c4 | x4114c0s1b0n0 | 171.1 | 168.5 | 157.2 | 161.0 | 161.7 | 102.3 | 108.2 | 102.6 | 99695-121670 |
  | c5 | x4114c0s1b0n0 | 171.3 | 168.7 | 157.3 | 160.6 | 162.2 | 102.3 | 108.4 | 102.7 | 121685-131060 + Fig. 3 |
  Start vs end anchors within a chunk: median |drift| 0.26%, max 2.6% (c3, k=16384). Across nodes: 4.4-7.0% spread per anchor (same as E5).
  At each seam the two chunks' nodes agree at the k that matter (c1/c2 at 8192/16384 within 0.7%; c2..c5 at 32768-98563 within 1%), so the
  merged curve has no node step; c1's node is ~6% slower only at k >= 32768, which c1 did not measure (except its anchors).
- **Steady state vs the original CSV (median over k of per-k medians, same k)**:
  | k range | odd | 2 mod 4 | 4 mod 8 | 8 mod 16 | 16 mod 32 | 32 mod 64 | 0 mod 64 |
  |---|---|---|---|---|---|---|---|
  | 64-1023 | 17.4 (1.00) | 62.2 (1.01) | 60.8 (1.02) | 62.8 (1.02) | 67.7 (1.02) | 76.2 (1.01) | 72.6 (1.02) |
  | 1024-4095 | 75.5 (0.89) | 118.7 (0.92) | 141.9 (0.93) | 146.1 (0.93) | 152.3 (0.94) | 161.7 (0.93) | 163.5 (0.94) |
  | 4096-16384 | 104.7 (0.91) | 111.1 (0.79) | 134.9 (0.80) | 139.8 (0.80) | 148.1 (0.79) | 159.3 (0.79) | 160.0 (0.80) |
  | 16385-24297 | 103.4 (0.97) | 103.5 (0.85) | 140.4 (0.83) | 146.9 (0.84) | 154.3 (0.83) | 162.1 (0.82) | 162.8 (0.83) |
  | 24298-40999 | 102.7 (1.01) | 116.9 (0.97) | 138.4 (0.91) | 143.9 (0.91) | 148.0 (0.89) | 157.4 (0.86) | 158.7 (0.86) |
  | 41000-131071 | 102.6 (1.00) | 124.6 (1.00) | 127.7 (0.99) | 131.6 (0.99) | 141.8 (1.00) | 160.5 (1.00) | 161.0 (1.00) |
  (new TFLOP/s, ratio new/old in brackets.) **The old ~200 "L2-resident" plateau is gone**: aligned k sustain ~158-163 TFLOP/s from k ~ 4k to 131k.
  The original numbers are reproduced exactly (ratio 1.00) where its runs were long (k >= 41k) or latency-bound (k < 1024); in between they were
  8-21% optimistic (post-idle boost, E4). Remaining real structure: (1) the 0/32-mod-64 classes are flat; (2) the 4/8/16-mod-32 classes step
  down by ~8-12 TFLOP/s between k ~ 24k and 41k (140-154 -> 128-142), i.e. a residual L2 -> HBM effect that hits only rows < 64 B aligned
  (consistent with the prior analysis's cache-line-splitting hypothesis; E8 targets it); (3) wave sawtooth with peaks at multiples of 1792 (dotted lines)
  for k < ~16k; (4) the k = 2 mod 4 square wave (period 1792) is present from k ~ 5k, not only above 30k (95/115 TFLOP/s for k < 16k).
- **k-slicing square wave over the whole streaming range** (k = 2 mod 4, k >= 40,960, step 15 grid, 1,502 points with predicted entry 463):
  levels 108.2 / 125.2, **replay agreement 100.00%**.
- **Clock/power** (tile 0, 500 ms): k < 1024: 1500 MHz, 162 W, never throttled; 1024-4095: 1300 MHz, 199 W, throttled 10% of samples;
  k >= 4096 on every node: **215.7 W**, throttled ("Average Power Excursion") 88-97% of samples, reported 1000-1100 MHz (a 500 ms average over
  alternating kernel types; not directly comparable to E4c's 850 MHz for a single repeated shape).
- **Measurement quality**: gap between points median 0.46 ms (p99 0.5 ms, max 103 ms once); warm-up median / measured median = 1.000 (median),
  p1 0.78 / p99 1.06. p10-p90 spread: 1.8% (k < 1024), **40% (1024-4095)**, 6.9% (4096-16384), 2.7% (k >= 41k). The 1024-4095 points last only
  25-75 ms each with sync gaps between calls, so the power state moves within a point; their medians are the least stable part of the curve.
- **Fig. 3 (m = k, n = 4096)**:
  | m=k | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 | 131072 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | new median | 0.18 | 0.70 | 2.84 | 11.3 | 37.3 | 79.9 | 216.6 | 203.4 | 168.9 | 169.3 | 169.1 | 169.0 | 169.1 |
  | original | 0.17 | 0.69 | 2.72 | 10.3 | 36.9 | 86.2 | 218.6 | 205.2 | 172.2 | 172.1 | 170.9 | 170.6 | 169.5 |
  | 250-call run (s) | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.03 | 0.04 | 0.17 | 0.81 | 3.2 | 13 | 52 | 208 |
  m = k >= 8192 sustain 169 (original within 0.2-2%: its runs were long). **m = k = 2048 and 4096 stay at 203-217 even in this protocol** because
  each point lasts only 40-170 ms after ~70 ms of tiny GEMMs (low duty cycle), i.e. they run in the boost window; their p10-p90 bands are wide
  (2048: 137-218; 4096: 166-240). A sustained number for those two sizes needs a longer run (>= 0.5 s, e.g. 3,000+ calls) — not done.
- **Verdict**: E9 delivered. New Figs. 2/3 show a single sustained regime (~160 TFLOP/s for aligned k on one stack) plus alignment/wave/k-slicing
  structure; the old two-regime picture was a measurement artefact for 4k <= k < 41k.

## E10 — All 12 stacks at once: is the ~215 W cap per stack or shared per GPU? (`runs/e10_20261026_053256/`, job 8966373, node x4118c0s1b0n0, 8.6 min)
- **Design** (`experiments/e10_allstacks.sh`, `bench/steady_run.py`, `analysis/e10_analysis.py` -> `e10_summary.txt`, `e10_table.csv`): 20 s of
  back-to-back `torch.mm` (2048 x 2048 x k) per stack, all processes released at a common wall-clock time, 20 s idle between phases;
  xpu-smi per tile at 100 ms (12 pty-wrapped loggers; stack s = GPU s//2, tile s%2 — confirmed: in p1 only GPU 0 tile 0 rises above the
  86 W idle level). Same-node single-stack baselines added (p1, p6), plus 2 stacks on one GPU (p2), 2 stacks on 2 GPUs (p3) and one stack per GPU (p4).
- **Measured** (steady = median TFLOP/s over 2-20 s; tile power/clock = median over 2-20 s):
  | phase | k | active stacks | TFLOP/s per active stack (median, range) | tile W | GPU W (both tiles) | MHz | throttle reason |
  |---|---|---|---|---|---|---|---|
  | p1 | 16128 | 0 | 167.7 | 215.7 | 302 (other tile idle 86 W) | 850 | Average Power |
  | p2 | 16128 | 0, 1 (same GPU) | 166.9 (165.6-168.2) | 201-204 | **404** | 850 | Current / Avg Power |
  | p3 | 16128 | 0, 2 (two GPUs) | 174.0 (168.2-179.8) | 215.7 | 302 / 303 | 850-900 | Average Power |
  | p4 | 16128 | 0, 2, ..., 10 (one per GPU) | 171.6 (167.8-180.3) | 215.6-215.9 | 299-305 | 850-900 | Average Power |
  | p5 | 16128 | all 12 | 166.6 (159.8-170.8) | 198-210 | **404-411** | 800-850 | Current Excursion (tile 0s) / + Avg Power (tile 1s) |
  | p6 | 98562 | 0 | 105.4 | 215.5 | 302 | 1000 | Average Power |
  | p7 | 98562 | all 12 | 99.8 (91.0-105.1) | 186-198 | **382-388** | 900-950 | Current Excursion / + Avg Power |
  Same stacks, p4 (one per GPU) -> p5 (all 12): stack 0 167.8 -> 165.7, 2: 180.3 -> 170.2, 4: 177.2 -> 167.5, 6: 172.0 -> 162.9,
  8: 170.9 -> 162.8, 10: 171.2 -> 169.1 (-1.2% to -5.6%). k = 98562: 105.4 alone -> median 99.8 with all 12 (-5%; worst stack 91.0, -14%).
  The post-idle boost is unchanged by concurrency: first 0.3 s at 198-210 TFLOP/s (k=16128) / 120-127 (k=98562) on every stack in p5/p7.
- **Against the earlier single-stack numbers**: E4c (other node) k=16128 sustained 171.4 at 850 MHz / 216 W — p1 here 167.7 at 850 MHz / 215.7 W.
  E5 per-stack spread 4-6% — p4's one-per-GPU stacks span 167.8-180.3 (7%) on this node.
- **Verdict**: the **~215 W "Average Power" limit is per stack (tile)**: one stack per GPU, each holds 215.7 W regardless of what other GPUs do.
  **There is an additional per-GPU limit**: with both stacks of a GPU busy, the GPU settles at **~404-411 W (k=16128) / ~385 W (k=98562)**,
  the throttle reason switches to **Current Excursion**, and each tile gets ~200 W instead of 216 W. Cost: 1-6% (k=16128) and ~5% median,
  up to 14% (k=98562) vs. running alone. Consequence: the original transformer sweeps (12 concurrent processes, one per stack) sit a few % below
  single-stack numbers in steady state, on top of the boost effect (E11a).

## E11a — Boost-contaminated points in the original transformer sweeps (no GPU; `e11a/`, `analysis/e11a_boost_flags.py`, `analysis/e11a_rerun_plan.py`)
- **Inputs (read-only)**: `sizing/scripts/median_csv/results_{h,s}_sweep_{blocks,layer,flash}_median/*.csv` (62,253 GEMM/layer points after removing
  repeated rows). Protocol from `sizing/scripts/batch_transformer*.sh`: **100 measured + 50 warm-up = 150 calls** per GEMM (not 250), median of the
  100 measured; operands created with CPU `randn` right before each GEMM (an idle gap that refills the boost); **12 processes at once, one per
  stack** (E10: costs 1-6% in steady state). FLOPs per GEMM from `transformer_flops.py`/`utils.py` (qkv 2sbh·3h, proj 2sbh², h->4h / 4h->h 2sbh·4h,
  logit 2sbvh, score/AOV 2b·s²·h, sdpa/flash 4b·s²·h); per-call time = FLOPs / reported TFLOP/s; run length = 150 x per-call time.
- **Flags** (`e11a/e11a_points_flagged.csv`): `flag_run_lt_0p5s`, and `f_boost` = share of the 100 measured calls that start inside a 0.35 s
  post-idle window (E4b/E4c) -> class `median_in_boost` (f >= 0.5), `partial` (0 < f < 0.5), `steady`.
  **23.0% of all points ran < 0.5 s; 23.9% have their median inside the boost window, 14.0% partially.** **All 1,151 points above 185 TFLOP/s
  are short-run (< 0.5 s) points**; no point with a run >= 0.5 s exceeds 185.
  | sweep | GEMM | n | run < 0.5 s | median in boost | partial | median run (s) |
  |---|---|---|---|---|---|---|
  | h_sweep_blocks | attention_score | 6,091 | 51% | 53% | 35% | 0.49 |
  | h_sweep_blocks | attention_over_value | 6,088 | 42% | 45% | 33% | 0.58 |
  | h_sweep_blocks | attention_linear_projection | 6,088 | 35% | 36% | 14% | 1.07 |
  | h_sweep_blocks | mlp_h_to_4h / mlp_4h_to_h | 6,087 each | 18% / 19% | 18% / 19% | 7% | 4.2 / 4.0 |
  | h_sweep_blocks | logit_block | 6,077 | 4% | 4% | 5% | 6.5 |
  | h_sweep_blocks | sdpa | 6,091 | 6% | 7% | 15% | 2.0 |
  | s_sweep_blocks | linear GEMMs | 1,916 each | 24-47% | 24-49% | 9-20% | 0.6-2.4 |
  | s_sweep_blocks | score / AOV | 1,916 each | 41% / 42% | 41% / 42% | 10% / 6% | 0.9 / 1.1 |
  | h/s_sweep_layer | full layer | 4,708 / 1,853 | 3% / 4% | 3% / 4% | 5% / 4% | 10.9 / 12.7 |
  | h/s_sweep_flash | flash_attn | 72 / 30 | 42% / 23% | 43% / 23% | 25% / 10% | 0.6 / 3.6 |
  Example (h sweep, a = 32, s = 2048): attention_linear_projection 190.2 TFLOP/s at h = 4096 (run 0.22 s) vs 168.2 at h = 8192 (run 0.98 s);
  medians are in the boost window for h <= 6112 (proj), <= 2976 (MLP), <= 544 (logit) and up to h ~ 12k for the score/AOV BMMs (short per call).
  Median TFLOP/s by class (blocks sweeps): proj 157.5 boosted vs 145.8 steady; h->4h 159.5 vs 148.4; 4h->h 170.4 vs 155.6; AOV 74.2 vs 44.4.
  (Logit and score have the opposite sign: their boosted points are the small, latency-bound ones.)
- **Proposed steady-state rerun (not run)** — `e11a/rerun_plan/` (`e11a_rerun_subset.csv` + 10 `gemm_bench.py` shape files):
  * subset: h sweep (s = 2048, mb = 4, v = 51,200) at a in {8, 32, 128} x h in {512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 2048 + a,
    8192 + a}; s sweep (a = 32) at s in {2048, 4096, 8192, 16384} x h in {1024, 2048, 4096, 8192}. All 5 linear GEMMs (F.linear, layout NT,
    m = s·mb tokens, shapes exactly as in the sizing benchmarks, incl. logit n = v, k = h) + score/AOV BMMs (layout NN, `--batch mb·a`). sdpa/flash and the
    full-layer sweeps are not GEMMs and need their own harness (the layer sweeps are only 3-4% short-run anyway).
  * 224 distinct shapes (135 NT, 89 batched NN); in the original data 88 are `median_in_boost`, 34 `partial`, 64 `steady` (controls), 38 have no
    `qkv_transform` value in its CSVs.
  * protocol: E6b style — each shape file starts with a heater line (16384 x 4096 x 16384, or a batched 2048 x 2048 x k with ~1 TFLOP per call,
    >= 1.5 s for 250 calls), shapes sorted large -> small, no idle between shapes, 50 + 200 calls (gemm_bench defaults).
  * **cost: ~24 min of GEMM time + ~3 min startup/heaters = ~27 min on one stack = ~0.45 node-hours** (estimate uses the original per-call times,
    deflated 20% for flagged points; largest single shape: logit at s = 16384, h = 8192, ~96 s). Given E10, run it on stack 0 alone, or on the 6
    even stacks (one per GPU, which E10 shows do not interact) concurrently for 6 replicas at the same node cost; avoid all 12 at once.
- **Verdict**: about a quarter of the transformer GEMM points (and essentially all values > 185 TFLOP/s) are boost-inflated; MLP/proj/AOV curves
  at small h and all BMMs at s = 2048 are the most affected. Rerun proposed, not executed.

## E8 — unitrace hardware metrics, k = 98560 (64 B rows) vs 98564 (8 B rows) (`runs/e8_20261026_054558/`, job 8966390, node x4500c6s6b0n0, 2 min)
- **Question**: from the prior analysis: within the 2D-block group the selected kernel is identical, so the 8 -> 64 B gain (~25%) is hardware —
  hypothesis "cache-line splitting of 64 B block rows".
- **Setup**: `unitrace -q -g <group>` (metric query per kernel instance; `perf_stream_paranoid = 0` on the compute node), groups found on the node:
  ComputeBasic, L1ProfileReads/Writes/SlmBankConflicts, L1Cache1-6; k = 98560 / 98564 plus 98568 (16 B) and 98576 (32 B); 10 + 20 calls,
  medians over the 20 measured kernel instances (`analysis/e8_parse.py` -> `e8_metrics_table.csv`, `e8_summary_full.txt`). The metrics are
  printed in `ut_<group>_k<k>.log` (unitrace 2.3.0 ignored `--output-dir-path` for them). Uninstrumented reference (`e8_timing.csv`, 400 calls):
  98560 166.3-167.2, 98564 128.4, 98568 132.5, 98576 143.6 TFLOP/s.
- **Measured per kernel call** (ComputeBasic unless noted; ratio vs 98560 in brackets):
  | metric | 98560 (64 B) | 98564 (8 B) | 98568 (16 B) | 98576 (32 B) |
  |---|---|---|---|---|
  | XMX instructions | 1.615e9 | (1.003) | (1.003) | (1.003) |
  | threads dispatched | 1792 | 1792 | 1792 | 1792 |
  | L1 (load/store cache) bytes read (L1ProfileReads) | 42.2 GB | (1.003) | (1.003) | (1.003) |
  | SEND instructions | 5.75e7 | (1.03) | (1.03) | (1.03) |
  | **GPU core cycles** | 3.91e6 | **(1.60)** | (1.52) | (1.31) |
  | **HBM bytes read** | 404 MB (= B once + A) | **639 MB (1.58)** | 599 MB (1.48) | 514 MB (1.27) |
  | HBM bytes written | 400 MB (= C) | (1.04) | (1.04) | (1.04) |
  | **L3 bytes written (fills)** | 404 MB | **781 MB (1.93)** | (1.81) | (1.56) |
  | L3 bytes read | 6.46 GB | (1.06) | (1.05) | (1.03) |
  | XMX utilisation | 90.9% | 57.2% | 60.2% | 70.2% |
  | XVE stall | 6.5% | 37.0% | 33.6% | 24.6% |
  | load/store-cache input queue busy (L1Cache1-6, per Xe-core group) | 35-48% | 94-98% | 94-98% | 93-97% |
  | avg GPU clock during the call | 887 MHz | 1122 MHz | 1079 MHz | 1000 MHz |
  | GPU time | 4.40 ms | (1.27) | (1.23) | (1.16) |
- **Interpretation**: same kernel, same instruction mix, same bytes requested at L1 — but rows that are not 64 B aligned make the 2D block loads
  touch extra cache lines: **+58% HBM read traffic and +93% L3 fill traffic at 8 B alignment** (+48% / +27% HBM at 16 B / 32 B), the
  load/store-cache input saturates (~95% vs ~40%), XVE stalls rise from 6.5% to 37%, and the kernel needs **60% more cycles**. Under the power
  cap the stall-bound kernel draws less power and clocks ~26% higher (1122 vs 887 MHz), so the wall-time penalty shrinks to 27% — matching the
  E9 steady-state class levels (0 mod 32 ~161 vs 4 mod 8 ~128). This also explains why the 4/8/16-mod-32 classes in E9 lose ~8-12 TFLOP/s when
  k crosses ~24k-41k: the extra line fetches are served by L3 while B fits there and by HBM afterwards.
- **Verdict: hypothesis confirmed** (cache-line splitting -> extra L3/HBM traffic and load/store stalls, not a different kernel). Caveat: 30-call
  runs, so absolute clocks are not steady-state; the counts (bytes, cycles, instructions) are clock-independent.

---
# Round 3 (2026-10-26): attention analysis — full write-up in `ATTENTION.md`
- **Runs**: `runs/attn_smoke_20261026_155216/` (5.7 min), `runs/attn_full_20261026_161022/` (25.5 min),
  `runs/attn_conc_20261026_164406/` (4.4 min).
- **Kernels on Aurora today**: `F.scaled_dot_product_attention` → oneDNN Graph `sdp` → AOT `micro_sdpa` (Q/S in SLM, 26-107 KB per
  work-group, 8-32 sub-groups depending on d, no spills); the sizing benchmarks' "Triton fused attention" = ringX's Triton FA-2 tutorial kernel
  (all in registers, no SLM, autotuned BLOCK_M=128/BLOCK_N=32/8 warps). No FA-3/4-style kernel; `flash_attn` absent.
- **Steady state, one stack** (b=4, a=32, s=2048, non-causal), d = 32/64/128/256: SDPA 48.6/80.7/88.0/96.6 TFLOP/s; best Triton tile
  60.1/80.1/77.0/28.4; ringX autotuned —/78.0/72.4/28.4 (70.6/66.0/27.0 with its permute copies). Per-clock XMX fraction 0.15-0.37 (SDPA),
  0.19-0.28 (Triton, d <= 128), vs 0.88 for the same-process GEMM.
- **Hardware**: DPAS 100% of 4096 FLOP/clk/Xe-core from registers (70-90% re-reading operands from L1); exp2 41.6/clk/Xe-core; FMA
  251 FLOP/clk; clocks under the 215 W cap: GEMM 850 MHz, DPAS 1.05-1.1 GHz, attention 1.15-1.5 GHz; 12-stack concurrency −3-5%.
- **Mechanism**: d <= 64 softmax-bound with no DPAS/softmax overlap (no-overlap model with the measured exp rate matches the L1-resident
  inner loop: 0.20/0.28/0.19 per clock at d = 32/64/128); d >= 128 register-bound (every config over 16 KB per thread spills; d=256
  always spills; forced small GRF 2.1-6.1x slower; oneDNN's SLM staging is 3.4x faster at d=256).
- **Original data**: Triton d=64 and d=256 reproduced (70.8 vs 70.6; 25.6 vs 27.0); its Triton d=128 (50.0) and all its SDPA values
  (27-44% lower) are **not** reproduced and are not explained by boost or concurrency — probably the ringX version (repo re-cloned
  2026-10-22) or the older driver (+42 vs +77). Recommend rerunning Fig. 7/8 with `attn/attn_bench.py`.
