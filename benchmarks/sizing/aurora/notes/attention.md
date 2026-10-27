# Attention on Aurora's Max 1550 — measured kernels, hardware limits, and a design from first principles

*Measured 2026-10-26 on Aurora compute nodes (one Max 1550 stack, FLAT mode, `ZE_AFFINITY_MASK=0`), frameworks/2025.3.1:
torch 2.10.0a0+git449b176, oneDNN v3.10.2, Triton 3.6.0 (Intel XPU backend), driver 1.6.33578+77. This replaces the
literature-only draft "Attention kernel design Intel Max 1550.md" wherever the two disagree (Section 7 lists every change).
Run directories (raw outputs on Aurora): `runs/attn_smoke_20261026_155216/`, `runs/attn_full_20261026_161022/` (main),
`runs/attn_conc_20261026_164406/`; processed outputs in `../data/attn/`; GEMM and clock results in `gemm_notebook.md` (E0-E10).
"Original sweep" means the September 2026 sizing sweeps behind the first draft's Figs. 7 and 8.*

## 0. Bottom line

1. **Two attention kernels actually run on Aurora today, and both are FlashAttention-2 designs.**
   - `F.scaled_dot_product_attention` goes through oneDNN Graph's fused `sdp` partition to the AOT kernel **`micro_sdpa`**, which stages Q and S in SLM (26-107 KB per work-group).
   - The sizing benchmarks' "Triton fused attention" is the **upstream Triton FA-2 tutorial kernel** (ringX `fused_attention.py`), which keeps everything in registers and uses no SLM.
   - No FA-3/FA-4-style kernel exists for this GPU.
2. **Steady state, one stack, b=4, a=32, s=2048, non-causal**:
   - SDPA reaches **80.7 / 88.0 / 96.6 TFLOP/s at d = 64 / 128 / 256**.
   - The best Triton tile reaches **80.1 / 77.0 / 28.4**, and the autotuned ringX path **78.0 / 72.4 / 28.4** (70.6 / 66.0 / 27.0 including ringX's layout copies).
   - Per clock, that is 28-37% of the XMX peak for SDPA and 19-28% for Triton (8% at d=256). The same-process GEMM reaches 88%.
3. **The hardware itself is not the ceiling.**
   - DPAS sustains **100% of 4096 BF16 FLOP/clk/Xe-core** when its operands are in registers.
   - What limits throughput is the **215 W per-stack power cap**, which sets the clock per workload: 850 MHz for GEMM, 1.05-1.1 GHz for pure DPAS, 1.15-1.5 GHz for attention, 1.5 GHz for vector code.
   - On top of that comes a **~405 W per-GPU** limit when both stacks are busy. It costs attention only 3-5%.
4. **At d <= 64 the softmax is the limit, and it does not overlap with DPAS.**
   - Measured exp2 throughput is **41.6 per clock per Xe-core** (FP32 FMA: 251 FLOP/clk).
   - A no-overlap model with this rate and ~5 vector ops per score element predicts **0.20 / 0.28 / 0.19** of XMX peak at d = 32 / 64 / 128. The FA-2 inner loop with K/V held in L1 measures **0.20 / 0.28 / 0.19**.
   - The real Triton kernel sits on the same curve even with two threads per vector engine.
5. **At d >= 128 the 16 KB register file per thread is the limit.**
   - Large-GRF mode is mandatory: forcing small GRF spills 9-21 KB and is 2.1-6.1x slower.
   - In large-GRF mode Triton still spills at d=128 whenever BLOCK_N >= 64, and in every one of the 27 configurations at d=256 (14-16 KB).
   - oneDNN avoids this by putting S/P in SLM, which is how it is **3.4x faster at d=256**.
6. **Remaining headroom.**
   - At d=64 the existing kernels are at the softmax bound, so the only lever is fewer vector instructions per score.
   - At d=128, a spill-free kernel with the measured DPAS-from-L1 efficiency should reach about 0.37-0.41 of peak per clock, ~100-115 TFLOP/s (derived).
   - At d=256, a register-feasible Triton design could match SDPA's ~97.
   - If the softmax were free, the ceiling would be the GEMM ceiling: ~170 TFLOP/s per stack, sustained.

## 1. Setup, protocol and provenance

- **Protocol.** Every point first ran 3 calls (compile or autotune), then a 0.5 s settle with no measurement, then at least 0.5 s and 10 timed calls (event pair plus sync per call); the median is reported.
  - The settle drains the ~0.3 s post-idle power boost found in E4.
  - A >= 2 s GEMM heater ran at the start of each process, and the GEMM ceiling was re-measured at the start and end of each one: 170.8-171.9 TFLOP/s at 850 MHz and 215 W, stable to 0.6%.
  - Clock and power come from xpu-smi on tile 0 at 100 ms, aligned to each measurement window.
  - **Every attention measurement sat at the 215 W cap** (215 ± 2 W, except d=256 Triton at 187-206 W).
- **FLOPs.** 4·b·a·s²·d, halved for causal: the FlashAttention convention, which is also the formula in the sizing benchmarks.
- **Kernels measured.**
  - (a) SDPA.
  - (b) ringX `fused_attention` with its own autotuner, on head-first tensors and with the `.permute().contiguous()` copies that ringX's `local_attn_forward` adds.
  - (c) The same Triton kernel copied verbatim into `attn/tri_fa_kernels.py`, with only the autotune decorator removed, swept over 81 tile configurations and follow-ups.
  - (d) Triton microbenchmarks: FMA, exp2/exp, DPAS, and the FA-2 inner loop with K/V in L1, with and without softmax.
- **Provenance caveats.**
  - The ringX repo was re-cloned on 2026-10-22 (branch `users/khalid/fused_attention_2`, HEAD `007f8cd`, May 2026), after the September flash sweeps. Its current `__init__` no longer exports `local_attn_forward`, which Megatron imports. **The exact kernel version behind the original Fig. 7 data is therefore not recoverable from the filesystem**; this branch is the closest candidate.
  - Those sweeps also ran on driver build +42; these runs used +77 (Triton kernels are compiled by the driver's compiler at load time).

## 2. Hardware properties that matter for attention (measured on Aurora unless marked)

| Property | Value | Source |
|---|---|---|
| Device hierarchy | FLAT: 12 devices per node, one per stack | E0 |
| Xe-cores per stack | **56** (448 EUs, 56 sub-slices in the device query) | E0 |
| XMX BF16 rate | **4096 FLOP/clk/Xe-core sustained** (DPAS with operands in registers: 4093-4115 measured) → peak = 229.4 TFLOP/s per GHz per stack (367 at 1.6 GHz) | attn micro |
| DPAS with operands re-read from L1 each step | 70-90% of XMX per clock (64x64x32: 70%; 128x128x32: 90%) | attn micro |
| Sustained clock under the 215 W stack cap | GEMM 850 MHz (172 TFLOP/s = 0.88 of peak per clock); DPAS-only 1.05-1.1 GHz (242-252 TFLOP/s); attention 1.15-1.5 GHz; FMA-only 1.5 GHz | attn micro/full, E4c |
| Power limits | ~215 W per stack sustained ("Average Power Excursion"); ~405 W per GPU when both stacks are busy ("Current Excursion"); ~270 W per stack for ~0.3 s after an idle gap (boost) | E4c, E10, attn_conc |
| FP32 vector FMA | 251 FLOP/clk/Xe-core (spec 256) | attn micro |
| exp2 / exp throughput (Triton `tl.math.exp2` / `tl.exp`) | **41.6 / 40.4 per clk per Xe-core** (5.2 per vector engine) | attn micro |
| Sub-group width used by both kernels | SIMD16 | unitrace |
| Register file per hardware thread | 256 GRF x 64 B = 16 KB in large-GRF mode (Triton's default recompiles there whenever spills > 1 KB; every FA kernel we built ended in 256-GRF mode) | Triton metadata |
| SLM | `micro_sdpa`: 26.5 KB (d <= 32), 52.5 KB (48-64), ~100-107 KB (d >= 80) per work-group; Triton: 0 | unitrace |
| Persistent GEMM wave | 56 work-groups per stack (one per Xe-core); k-period 1792 = 7 x 256 | E2, E3, E9 |
| Row alignment for 2D block loads | 64 B rows needed; 8 B rows cost +58% HBM reads, +60% cycles for the same kernel | E8 |
| Tile quantization in the AOV BMM | Period 256 in d (tile height 256) | E7 |

## 3. The kernels that exist, and what they do

**oneDNN `micro_sdpa`**: reached via `F.scaled_dot_product_attention`, then the oneDNN Graph partition `sdp: matmul_qk; scale_mul; softmax; matmul_v` (causal adds a mask chain), then the `sdp_primitive_kernel_t` implementation.

| d | work-group (sub-groups) | SLM per work-group | grid (Q tiles x heads x batch) | Q-tile rows | spills |
|---|---|---|---|---|---|
| 16, 32 | 8 | 26.5 KB | 16 x 32 x 4 | 128 | 0 |
| 48, 64 | 16 | 52.5 KB | 16 x 32 x 4 | 128 | 0 |
| 80, 96, 128 | 32 | 104.5 KB | 16 x 32 x 4 | 128 | 0 |
| 160-256 | 32 | 100.5 KB | 16 x 32 x 4 | 128 | 0 |
| 384, 512 | 32 | 98.3 KB | 32 x 32 x 4 | 64 | 0 |

- From d >= 80 a work-group fills an Xe-core: 32 threads at 4 per vector engine in large-GRF mode, plus >= 100 KB of SLM. So a wave is 56 work-groups and 2,048 tiles take 36.6 waves (98.8% wave efficiency).

**Triton FA-2 forward** (ringX `fused_attention.py`, derived from the Triton tutorial):
- Structure:
  - One program per (Q tile, batch x head); not persistent.
  - Q, the fp32 O accumulator, S/P and the running max/sum all live in registers; no SLM.
  - K/V are read through device-side tensor descriptors (2D block loads).
  - Online softmax in base 2.
- Constraints: head dims {16, 32, 64, 128, 256}, sequence length a multiple of 128.
- Autotune space: BLOCK_M {64, 128} x BLOCK_N {32, 64, 128} x stages {2, 3, 4} x warps {4, 8, 16}; 57 s (d=64), 158 s (d=128) and 310 s (d=256) of autotuning per new shape key.
- The autotuner chose BLOCK_M=128, BLOCK_N=32, 8 warps (16 at d=256).

**Present but not the default path:** IPEX 2.10.10 (archived upstream; importing it also pulls in DeepSpeed), and ringX's "portable" backend (fp32 bmm + softmax). **Absent:** the `flash_attn` package and any FA-3/FA-4-style kernel.

## 4. Measured throughput (steady state, one stack, b = 4, bf16)

**Head-dim sweep** (a = 32, s = 2048, non-causal; TFLOP/s, with the fraction of XMX peak at the measured clock in brackets):

| d | SDPA (oneDNN) | Triton, best tile (config) | Triton via ringX autotuner | ringX incl. permute copies | original CSV: Triton / SDPA |
|---|---|---|---|---|---|
| 16 | 26.0 [0.08] | n/a (no valid tile) | — | — | 18.4 / 16.0 |
| 32 | 48.6 [0.15] | 60.1 [0.19] (256x32, 8 warps) | — | — | 50.4 / 29.7 |
| 48 | 56.2 [0.19] | — | — | — | — / 40.2 |
| 64 | 80.7 [0.28] | 80.1 [0.28] (256x32, 16 warps) | 78.0 | 70.6 | 70.8 / 58.8 |
| 80 | 62.9 [0.20] | — | — | — | — / 35.4 |
| 96 | 75.0 [0.26] | — | — | — | — / 39.2 |
| 112 | 71.8 [0.24] | — | — | — | — |
| 128 | 88.0 [0.32] | 77.0 [0.25] (256x32, 16 warps) | 72.4 | 66.0 | 50.0 / 57.1 |
| 144 | 64.2 [0.21] | — | — | — | — |
| 160 | 72.7 [0.24] | — | — | — | — / 36.6 |
| 192 | 79.1 [0.27] | — | — | — | — / 48.1 |
| 224 | 84.5 [0.31] | — | — | — | — |
| 256 | **96.6** [0.37] | 28.4 [0.08] (256x32, 32 warps; spills 14 KB) | 28.4 | 27.0 | 25.6 / 55.4 |
| 384 | 70.2 [0.26] | — | — | — | — / 39.1 |
| 512 | 83.1 [0.35] | — | — | — | — / 44.4 |

- **Causal** (FLOPs halved): Triton runs at 0.72-0.84 of its non-causal throughput at s=2048 and 0.87-0.96 at s=8192; SDPA at 0.81 / 0.83 / 0.90 (d = 64 / 128 / 256).
- **Sequence length** (a = 32): SDPA d=128 rises from 88.0 (s=2048) to 94.0 (s=16384); d=64 is flat at 81-83. Triton at s=8192: 64.4 / 80.8 / 84.8 / 31.5 for d = 32 / 64 / 128 / 256.
- **Heads**: flat, a = 8-128 changes SDPA by <= 5%.
- **12 stacks at once** (same node, `attn_conc`): SDPA and Triton lose 3-5% (each tile drops to ~200-207 W under the per-GPU limit); Triton at d=256 loses nothing.
- **ringX's layout copies** (`bshd` to `bhsd` and back, inside the timed call) cost 4-14%.

## 5. Why: decomposing the forward pass

Fraction of XMX peak per clock (`runs/attn_full_*/attn_decomposition.txt`):

| d | DPAS only, operands in registers | Inner loop, K/V in L1, **no softmax** | Inner loop, K/V in L1, **with softmax** | No-overlap model (below) | Real Triton kernel, best tile | SDPA |
|---|---|---|---|---|---|---|
| 32 | 1.00 | 0.62 | 0.20 | 0.20 | 0.19 | 0.15 |
| 64 | 1.00 | 0.56 | 0.28 | 0.28 | 0.28 | 0.28 |
| 128 | 1.00 | 0.28* | 0.19* | 0.19 | 0.25 | 0.32 |
| 256 | 1.00 | 0.07* | 0.08* | 0.06 | 0.08 | 0.37 |

\* Spilling microkernel configurations (128x64 tile, 8-16 warps); at d=128 the real kernel's best tile (BLOCK_N=32) avoids the spills.

**Model: softmax and DPAS run back to back, not overlapped.**
- Per score element, the matmuls cost 4d FLOP, plus one exp2 and about V ≈ 5 other vector operations (row max, scale-and-subtract, row sum, bf16 convert, rescale) on 16 lanes x 8 vector engines.
- The softmax-to-matmul time ratio is ρ = 1024/(d·X) + 8V/d. With the measured **X = 41.6** this gives **ρ = 2.0 / 1.0 / 0.5 / 0.25** at d = 32 / 64 / 128 / 256.
- If nothing overlaps, utilization = (no-softmax efficiency)/(1 + ρ). This reproduces the L1-resident loop to within 0.01 at d <= 128.
- V ≈ 5 is assumed, not counted from the ISA. Only X is measured.
- **The real kernel sits on the same curve with 16 warps (two threads per vector engine)**, so in the code Triton generates today a second thread does not hide the softmax behind the first thread's DPAS.

**Registers.**
- The per-thread budget is S (4·Rq·Bc) + P (2·Rq·Bc) + O (4·Rq·d) + m, l (8·Rq) + Q (2·Rq·d), where Rq = BLOCK_M / warps.
- Every Triton config whose budget exceeds 16 KB spilled (21 of 21). Some small-Rq configs at d=256 spill below it too.
- Measured examples:
  - d=128 at Rq=16, Bc=32 fits with Q in registers (no spills, 77.0).
  - Bc >= 64 at d=128 spills (max 44 TFLOP/s).
  - d=256 always spills.
- Forcing small GRF (8 KB per thread) spills 8.8-20.9 KB and is 2.1-6.1x slower.

**Rows per sub-group** (operand reuse): best non-spilling throughput by Rq = 2 / 4 / 8 / 16 / 32:
- d=64: 19.6 / 39.0 / 77.4 / 80.1 / —.
- d=128: 16.2 / 33.8 / 65.5 / 77.0 / spills.
- Each K/V B-fragment feeds Rq/8 DPAS instructions, so Rq >= 8 (ideally 16) is required. This matches the L1-intensity argument in the earlier draft.

**Other knobs.** num_stages (2/3/4) changes throughput by <= 2%. At d=32, the d-axis tile is 32 and BLOCK_N <= d, so S/P tiles are tiny and the softmax dominates (ρ = 2).

**SDPA's head-dim sensitivity.**
- Non-multiples of 32 lose 10-20% against their neighbours: d=80 62.9 vs 80.7 (64) and 75.0 (96); d=144 64.2 vs 88.0 (128) and 72.7 (160); d=384 70.2 vs 96.6 (256) and 83.1 (512).
- The work-group and SLM configuration also changes at d = 48 / 80 / 384 (Section 3).
- Hypothesis (not verified): padding of the head-dim tile inside the microkernel.

## 6. What this means for the paper's attention statements (Fig. 7 / 8)

- **"Triton fused kernel peaks at d = 64": confirmed in steady state on one stack** (80.1 vs 60.1 at d=32 and 77.0 at d=128). The mechanism is now measured, not hypothesised: softmax-bound below 64, register-bound above.
- **The size of the drop at d=128 in the original data is not reproduced.** It has 50.0 (and 47-49 at a >= 40); the same kernel family gives 66.0 via ringX (with copies) and 72-77 without. Possible causes:
  - an older ringX version that picked a BLOCK_N >= 64 tile, which spills here at 42-44 TFLOP/s;
  - the older driver.
  - Concurrency is ruled out (−3%), and the boost would push numbers up, not down.
- **d=256 matches** (25.6 vs 27.0).
- **SDPA in the original CSVs is 27-44% below today's single-stack numbers** at every d (e.g. 57.1 vs 88.0 at d=128), and 25-40% below today's 12-stack numbers. Neither the boost nor concurrency explains it; the software or driver state of the September runs is the remaining candidate. **Recommendation: rerun Fig. 7/8 with this protocol** (`attn/attn_bench.py sdpa|ringx`, ~15 min per sweep).
- **"SDPA is sensitive to head-dim divisibility": confirmed** (Section 5).
- **"Wave period d_h = 256"**: for the AOV BMM this is tile quantization with a 256-wide tile (E7: peaks at 256 and 512), not wave quantization. In the fused kernels d is not a tiled dimension at all; their d-dependence comes from Section 5.
- **Efficiency denominator.** Report attention as a fraction of XMX peak at the measured clock (or against a same-process GEMM, 172 TFLOP/s at 850 MHz). Do not use 367/734 at 1.6 GHz: the power cap sets a different clock for each kernel.
  - Against the same-process GEMM: SDPA 0.47-0.56, Triton 0.45-0.47 (d <= 128).

## 7. Corrections to the earlier draft ("Attention kernel design Intel Max 1550.md")

| Draft statement | Measured on Aurora |
|---|---|
| 56 vs 64 Xe-cores per stack unverified; FLAT vs COMPOSITE unknown | **56** per stack, **FLAT** (12 devices per node) — confirmed |
| "Published GEMM reaches only ~59% of peak; a limiter common to all XMX types, more likely than the power cap" | The ceiling is the **power cap through the clock**: DPAS runs at 100% of peak per clock, GEMM at 88% per clock but at 850 MHz under 215 W/stack. Sustained GEMM is 157-172 TFLOP/s per stack (E9), not 216 |
| "~240 TFLOP/s GEMM exceeds the published 216" | 200-240 TFLOP/s values are the ~0.3 s post-idle boost (~270 W, 1.0-1.2 GHz), captured by short runs (E4) |
| exp rate unknown (X = 32 assumed) | **X = 41.6** exp2/clk/Xe-core (exp: 40.4) |
| Softmax competes with XMX at small d (hypothesis) | **Confirmed**: no-overlap model matches within 0.01 at d <= 128 |
| d >= 128 exhausts registers (hypothesis); large GRF "effectively mandatory" | **Confirmed**: spills at d=128 for Bc >= 64 and at d=256 always; forcing small GRF is 2.1-6.1x slower |
| Register reserve 3 KB (hypothesis) | Rq=16, Bc=32, d=128 with Q in registers (15.5 KB by the formula) does **not** spill, so the effective reserve is under ~0.9 KB, or P aliases S |
| Two threads per vector engine needed to overlap softmax with DPAS (hypothesis) | With Triton's code, 16 warps (two threads per vector engine) **do not** overlap them (the kernel stays on the no-overlap curve) |
| Wave period 2048 vs 1792 open (64 vs 56 cores) | **1792 = 7 x 256** measured (E3, E9); wave = 56 |
| "Wave period d_h = 256" probably tile quantization | Confirmed for the BMM (E7) |
| GEMM saturation at 32 \| k = 64 B alignment (hypothesis); "16 B pitch rule predicts a cliff at multiples of 8" | 64 B rows confirmed as the full-speed case (E8); oneDNN requires >= 8 B rows for 2D block loads (4 \| k); k = 2 mod 4 uses fallback kernels and odd k the copy path (E1-E3) |
| Fig. 7 values read by eye (~55/70/52/27) | Original CSV: 50.4 / 70.8 / 50.0 / 25.6 (a=32); steady-state today: Triton 60.1 / 80.1 / 77.0 / 28.4, SDPA 48.6 / 80.7 / 88.0 / 96.6 |
| oneDNN SDPA tile and SLM sizes unknown | Section 3 table (unitrace) |
| L2 bandwidth only from a Max 1100 | Still not measured on Aurora (Section 9) |

## 8. Suggested attention-kernel design for Xe-HPC

Tags: **[M]** follows from a measurement here, **[D]** derived from measured inputs, **[H]** hypothesis still to test.

1. **Large-GRF mode, SIMD16 sub-groups** [M]. Small GRF spills at every d we built.
2. **Rq = 16 query rows per sub-group, BLOCK_N = 32, 16 sub-groups per work-group (BLOCK_M = 256)** at d = 64 and 128 [M]. This was the best measured tile; Rq < 8 halves throughput.
3. **Keep the per-thread register budget <= 16 KB with margin** [M/D]. At d = 128 this forbids BLOCK_N >= 64 with Q in registers. At d >= 256:
   - move P (bf16) to SLM and split the O accumulator over sub-groups by head-dim halves [D];
   - or, simplest, use oneDNN `micro_sdpa`, which already does this and gets 96.6 vs 28.4 [M].
   - Triton cannot address SLM explicitly, so a Triton kernel is the wrong tool at d >= 256.
4. **The softmax is additive on this stack, so minimise vector instructions per score** [M/D].
   - At d = 64, ρ ≈ 1 caps the kernel at about half of the no-softmax rate. Every vector op per score element removed is worth ~8/d of matmul time.
   - Candidates [H]:
     - fold scale and log2 e into one FMA before exp2 (already done);
     - FA-4-style conditional rescale, skipping `acc *= alpha` when the max did not grow; at d=128 that is BM·d multiplies per step;
     - packed-bf16 arithmetic for the max/sum;
     - fusing the bf16 convert into the exp2 output.
   - At d = 32 the softmax is 2x the matmul time. Pad d to 64 or group heads: that costs 2x the FLOPs but runs at ~1.3x the rate, so it is only worth it if ρ cannot be reduced.
5. **True DPAS/softmax overlap** [H]. The hardware runs DPAS on a separate XMX unit, but Triton's code shows no overlap even with two threads per vector engine.
   - Test first with a hand-written ESIMD/SYCL kernel: one thread issues DPAS on tile i while another runs the softmax of tile i−1.
   - If it works, the bound at d = 64 rises from 0.28 to the no-softmax 0.56 per clock (derived), i.e. ~2x.
6. **K/V path** [M]. DPAS that re-reads its operands from L1 runs at 70-90% per clock, versus 100% from registers. Larger B-fragment reuse (Rq = 16) and 2D block prefetch 2-3 steps ahead are the levers; num_stages 2-4 made no difference in Triton.
7. **Grid** [D]:
   - one work-group per (Q tile, head) as both kernels do; plan tile counts so b·a·s/BLOCK_M is a multiple of 56 x (work-groups per Xe-core);
   - for causal attention, launch the heaviest Q tiles first (Triton's causal path runs at 0.72-0.84 of non-causal at s = 2048).
8. **Normalisation** [M]:
   - report TFLOP/s together with the measured clock and the fraction of XMX peak at that clock, using >= 0.5 s steady-state windows;
   - report runs with both stacks of a GPU busy separately (−3-5%).
9. **Today's headroom** [D]:
   - d = 64: at the softmax bound (0.28 per clock); gains require item 4 or 5.
   - d = 128: a spill-free kernel at the measured no-softmax efficiency (0.56-0.62) and ρ = 0.5 should reach 0.37-0.41 per clock, ~100-115 TFLOP/s at ~1.2 GHz, versus 77 (Triton) and 88 (SDPA).
   - d = 256: register-feasible designs reach ~97 (SDPA).
   - Hard ceiling with a free softmax: the same-process GEMM, ~170 TFLOP/s per stack sustained.

## 9. Still unknown, and the experiments that would settle it

| Question | Experiment | Cost |
|---|---|---|
| Can DPAS and vector or extended-math work overlap across threads, or within one thread? | ESIMD microkernel: DPAS stream + exp2 stream, 1 vs 2 vs 4 threads per vector engine | ~1 day of coding + 0.1 node-h |
| Exact vector-op count V per score element | Dump the Triton kernel's ISA (`TRITON_INTEL_DUMP_SHADER_INFO` / native code) and count | 1 h, no GPU |
| L2-to-Xe-core and HBM bandwidth on Aurora | Streaming microbenchmark at 64 MB / 1 GB working sets | 0.1 node-h |
| Why the September SDPA / Triton d=128 numbers are 25-43% lower | The exact ringX commit and driver; rerun `benchmark_sdpa` / `benchmark_flash` under this protocol | 0.25 node-h |
| DPAS latency and repeat count | Dependent-chain ESIMD test | with the overlap test |
| Backward pass | Same decomposition for `_attn_bwd` and oneDNN SDPA backward | ~0.5 node-h |

## Files

`attn/attn_bench.py` (harness), `attn/tri_fa_kernels.py` (verbatim kernel copy), `attn/attn_id.py`, `attn/conc_run.py`,
`attn/analyze_attn.py`; payloads `experiments/attn_{smoke,full,conc}.sh`. Per-run outputs:
`attn_*.jsonl` (raw rows), `attn_all.csv` (with clock and power), `attn_summary.txt`, `attn_decomposition.txt`, `attn_register_model.txt`,
`attn_ids.csv` and `id_*.log` (unitrace + ONEDNN_VERBOSE), `attn_conc_summary.txt`, `smi_*.txt`.
