# GEMM and attention measurements on Aurora (Intel Data Center GPU Max 1550)

This directory holds the benchmarks, job scripts, analysis, data and figure scripts behind the steady-state results of the
paper *Performance Evaluation and Characterization of Large Language Model Workloads on the Aurora System*. Everything here
measures one GPU stack at a time in FLAT mode, after the stack has reached its sustained power state.

The sections below explain how each measurement works and how we found out which kernel ran and why it was fast or slow.
The experiment-by-experiment record, with the question, prediction and result of each run, is in
[`notes/gemm_notebook.md`](notes/gemm_notebook.md) and [`notes/attention.md`](notes/attention.md).

## Layout

| Path | Contents |
|---|---|
| `bench/` | GEMM benchmarks (`gemm_bench.py`, `steady_sweep.py`, `steady_run.py`), environment capture, `xpu-smi` logger (`smi_log.sh`) |
| `attn/` | Attention benchmarks and kernel identification (`attn_bench.py`, `attn_id.py`, `conc_run.py`), a copy of the ringX Triton kernel |
| `replay/` | Offline port of oneDNN v3.10.2 GEMM kernel selection (`gemm_predict.py`, `selector_sim.py`, `catalog.json`) |
| `experiments/` | One PBS payload per experiment (E0 to E10, attention) |
| `pbs/submit.sh` | Wraps a payload in a one-node PBS job |
| `analysis/` | Joins measurements with the replay, parses counters and power logs, fits the wave model |
| `data/` | Processed outputs of every experiment (CSV and text summaries) and a few raw logs as examples |
| `figures/` | Scripts that regenerate the paper figures from `data/` into `figures/out/` |
| `notes/` | Lab notebook for the GEMM experiments and the attention write-up |

## Quick start

The runs used the `frameworks/2025.3.1` module (PyTorch 2.10.0a0+git449b176, oneDNN v3.10.2 commit f1d47193, Triton 3.6.0
with the Intel XPU backend, GPU driver 1.6.33578+77). The module sets `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, so a node shows 12
devices, one per stack. `ZE_AFFINITY_MASK=<s>` selects stack `s`, which is tile `s % 2` of GPU `s // 2`.

```bash
module load frameworks
cd benchmarks/sizing/aurora

# one GEMM point, 50 warm-up and 200 timed calls (run on a compute node)
ZE_AFFINITY_MASK=0 python3 bench/gemm_bench.py --m 2048 --n 2048 --k 98560 98562 --out /tmp/test.csv

# which kernel oneDNN would pick, without a GPU
python3 -c "import sys; sys.path.insert(0, 'replay'); from gemm_predict import predict; print(predict(2048, 98562, 2048))"

# submit an experiment as a one-node job; output goes to runs/<name>_<timestamp>/
PBS_ACCOUNT=<your project> bash pbs/submit.sh e3 00:40:00 experiments/e3_repeat.sh debug

# regenerate the paper figures from data/
for f in figures/make_*.py; do python3 $f; done
```

`pbs/submit.sh` sources `$SETUP_SH` if it is set and otherwise loads `frameworks`. It exports `QA` (this directory) and `RUN`
(the run directory) to the payload. The `ringx` subcommand of `attn/attn_bench.py` also needs a ringX-attention checkout that
provides `ringX_attn.fused_attention` on `PYTHONPATH` (we used commit 007f8cd of the ALCF copy).

Notation follows `mm_flops.py`: `C[m x k] = A[m x n] @ B[n x k]`, so `n` is the inner dimension and `k` is the row length of
`B` and `C`. `--layout NT` is the `F.linear` forward, `NN` the input gradient and `TN` the weight gradient.

## Measurement protocol

### Why the protocol matters on this GPU

A stack may draw about 270 W for a few hundred milliseconds after it has been idle, and then the firmware holds its average
power at 215 W by lowering the clock to 800 to 850 MHz for GEMMs (E4c). When both stacks of a GPU are busy, a second limit
holds the GPU at about 405 W, and each stack gets about 200 W (E10). A benchmark that allocates new operands on the host
before each shape gives the stack an idle gap, and a short run after that gap measures the boost. The original sweeps of the
sizing benchmarks did this, and shapes whose 250 calls lasted less than about half a second came out 8 to 21% too high.

### GEMMs (`bench/steady_sweep.py`, used for the paper's K and M=K sweeps)

1. Allocate three flat device buffers once, large enough for the biggest point. Each point uses contiguous views of them, so
   the row stride equals the point's own `k` and nothing is allocated between points (the median gap is 0.46 ms).
2. Run a `(2048,2048) x (2048,16128)` GEMM back to back for at least 2 s.
3. Measure anchor shapes after the pre-heat and again at the end, to compare jobs and nodes.
4. For each point run 50 warm-up calls and 200 timed calls, each timed with an event pair and followed by a synchronize.
   Report the median call time. The fastest calls reflect short clock transients and are not used.

`bench/gemm_bench.py` is the general tool for the other GEMM experiments. Its options reproduce or isolate each effect.
`--init host` recreates the old host-side operand creation, `--idle_s` adds an idle gap, `--rotate_mb` cycles through enough
operand copies to defeat the 192 MiB L2, `--burst` launches several calls between timing events, `--dump_iters` writes the
duration and host start time of every call for alignment with power logs, and `--profile` writes a `torch.profiler` trace.

### Fused attention (`attn/attn_bench.py`)

Each point runs 3 calls to compile or autotune, 0.5 s without timing, and then at least 10 timed calls over at least 0.5 s.
We report the median. Each process starts with a 2 s GEMM heater and re-measures a GEMM at the start and the end as a ceiling.

### Power traces (`experiments/e4c_clock.sh`, `bench/steady_run.py`)

These runs start after 15 s of idle and time every call, to show the transition from the boost to the sustained state.

### Variation

Stacks differ by 4 to 6% in sustained throughput, and the ordering of stacks is the same for every shape (E5, E10). Compare
shapes on the same stack, or report the median of several stacks.

## How we probed the kernels

### 1. Which implementation oneDNN runs: `ONEDNN_VERBOSE`

```bash
ONEDNN_VERBOSE=profile_exec python3 bench/gemm_bench.py --m 2048 --n 2048 --k 98561 --iters 5 --warmup 2 --out /tmp/v.csv
```

Each GEMM prints one line with the implementation name, the shapes and the time in ms (examples in
`data/e1_e2/logs/e1_k*.log`). On this system `jit:gemm:any` is the no-copy gemmstone path, and `jit:xe_hp:gemm:any` is the
path that first copies the operands into an aligned layout. `ONEDNN_VERBOSE=all` also prints why other implementations were
skipped and the strategy string of the selected kernel, which identifies the catalog entry (`data/e1_e2/logs/e1_all.log`).
For attention, the same variable shows that `F.scaled_dot_product_attention` goes through a oneDNN Graph `sdp` partition to
the `micro_sdpa` kernel (`data/attn/id_logs/`).

### 2. Kernel names, launch grid and extra operations: `unitrace -d -v`

```bash
unitrace -d -v python3 bench/gemm_bench.py --m 2048 --n 2048 --k 98562 --iters 3 --warmup 1 --out /tmp/u.csv
```

The device timeline lists every kernel with its SIMD width, global and local size, arguments and SLM
(`data/e1_e2/logs/e2_unitrace_k*.log`). This is how we saw that the large GEMM kernels are persistent, with
`{56; 1; 1}` work-groups (one per Xe-core) of `{128; 4; 1}` work-items. It also showed two copy kernels in front of the GEMM
for odd `k`, and for `k = 98562` a `zeCommandListAppendMemoryFill` of 14,680,064 bytes before each call. That fill zeroes the
fp32 buffer that split-K (oneDNN's variable k-slicing) accumulates into, so it is a direct sign that split-K is on.
For Triton kernels, `attn/attn_id.py` prints the register count, spill size and GRF mode from the compiled kernel's metadata.

### 3. Hardware counters: `unitrace -q -g <group>`

```bash
unitrace --metric-list                      # groups available on the node
unitrace -q -g ComputeBasic python3 bench/gemm_bench.py --m 2048 --n 2048 --k 98564 --iters 20 --warmup 10 --out /tmp/c.csv
```

The metric query reports, per kernel instance, the XMX instruction count, cycles, bytes read from HBM, L3 fills, vector
engine stalls and load/store queue occupancy. Aurora compute nodes allow this without extra permissions
(`perf_stream_paranoid = 0`). `analysis/e8_parse.py` extracts the CSV blocks from the logs and takes medians over the measured
calls. E8 used this to show that one kernel at 8-byte row alignment executes the same instructions as at 64-byte alignment
but reads 58% more data from HBM and needs 60% more cycles (`data/e8/`, example logs in `data/e8/logs/`).

### 4. Clock, power and throttle reason: `xpu-smi dump`

```bash
bash bench/smi_log.sh smi.txt 0 0 100 &    # GPU 0, tile 0, 100 ms; wraps xpu-smi dump -m 0,1,2,3,35 --ims 100 --file
pkill -f "xpu-smi dump"                     # stop
```

Metrics 0, 1, 2, 3 and 35 are utilization, power, frequency, temperature and throttle reason. Two details are easy to miss.
Sampling faster than 1 s (`--ims`) works only together with `--file`, and `--file` mode needs a terminal, so under PBS it has
to run inside `script` with an input that stays open, which is what `bench/smi_log.sh` does. Log the tile that runs the work (`-t`), because the device-level
frequency averages both tiles. `--dump_iters` in `gemm_bench.py` and the host timestamps in the attention outputs let you
align each call with the power samples (`figures/make_power.py`, `analysis/e10_analysis.py`).

### 5. Predicting the kernel choice: the selector replay in `replay/`

oneDNN chooses among about 1,200 pre-tuned GEMM kernels (the catalog in `src/gpu/intel/gemm/jit/selector/db/kernel.db`). For
each candidate that the problem's layout and alignment allow, it estimates the run time with an analytic model and picks the
cheapest. The replay ports this to Python for bf16 on one stack (448 vector engines).

- `parse_db.py` turns `kernel.db` into `catalog.json` (`fetch_catalog.sh` downloads v3.10.2 and rebuilds it).
- `selector_sim.py` ports the derived kernel parameters and the cost model `evaluateE`, including variable k-slicing.
- `gemm_predict.py` maps a PyTorch matmul to oneDNN's column-major problem, applies the eligibility rules and the sort order
  (fallback kernels rank after kernels with compatible stricter alignment), and returns the selected entry, tile size, number
  of tiles and slots, wave and padding efficiency, whether k-slicing is on, or `copy_based=True` for the copy path.

```python
from gemm_predict import predict
predict(2048, 98562, 2048)                    # torch.mm(A[2048x2048], B[2048x98562]): entry 463, kv=True
predict(8192, 4096, 4096, transB_t=True)      # F.linear forward
```

`analysis/compare_replay.py <bench.csv>` joins a `gemm_bench.py` CSV with the predictions. The replay predicts whether split-K
is on for every steady-state point with even `k` not divisible by 4 above `k = 40,960` (1,502 points of E9). For one period
measured separately (E3) it agrees with 99.8% of the points when each point is judged by its fastest call and with all of
them when judged by the median (`data/e3/e3_replay_summary.txt`). `analysis/wave_fit.py` fits one throughput constant per kernel and uses the replay's
efficiencies to reproduce the aligned K sweep within 2% (the appendix table of fitted constants). The copy path is detected
but not modelled. Check predictions for new layouts against `ONEDNN_VERBOSE` before relying on them.

To port a newer oneDNN version, start from `src/gpu/intel/gemm/jit/selector/kernel_selector.cpp` and `kernel_evaluator.cpp`
and from `use_nocopy()` in `src/gpu/intel/gemm/jit_xe_hp_systolic.cpp` (the copy-path rule), then rerun
`compare_replay.py` on the E3 and E9 data.

### 6. Decomposing attention: microbenchmarks in `attn_bench.py micro`

The `micro` subcommand measures the FMA and `exp2` rates, DPAS throughput with operands in registers and re-read from L1, and
the FlashAttention-2 inner loop with K and V resident in L1, with and without the online softmax. Comparing the inner loop
with and without the softmax, against a model in which the two run one after the other, is how we found that the softmax
does not overlap with the matrix products at small head dimensions. `tricfg` sweeps tile sizes, warps and GRF mode of the
same Triton kernel and records spills, which shows where the 16 KB register file per thread runs out. Details and numbers are
in `notes/attention.md`.

## Experiments

| ID | Question | Payload | Data | Used in the paper |
|---|---|---|---|---|
| E0 | Software versions, device mode, original benchmark statistic | `e0_env.sh` | (notes only) | Experiments section |
| E1 | Which oneDNN implementation runs for each row alignment | `e1_verbose.sh` | `data/e1_e2/` | Kernel Selection |
| E2 | Kernel names, launch grid, copy kernels, split-K fill | `e2_trace.sh` | `data/e1_e2/` | GEMM Execution, Split-K |
| E3 | One 1792-wide period of the split-K oscillation, odd `k` | `e3_repeat.sh`, `e3b_odd.sh` | `data/e3/` | Split-K figure, appendix replay |
| E4 | L2 residency or run length behind the old 200 TFLOP/s plateau | `e4_l2_clock.sh`, `e4b_clock.sh` | `data/e4/`, `data/e4b/` | Power and Frequency, appendix |
| E4c | Clock, power and temperature from idle to sustained | `e4c_clock.sh` | `data/e4c/` | Power figure |
| E5 | Stack-to-stack spread on one node | `e5_stacks.sh` | `data/e5/` | Appendix stack table |
| E6 | Transformer-shaped GEMMs and vocabulary padding | `e6_linear.sh`, `e6b_sustained.sh` | `data/e6/`, `data/e6b/` | Transformer Results, appendix |
| E7 | Attention-over-values BMM against head dimension | `e7_bmm_dh.sh` | `data/e7/` | Transformer Results |
| E8 | Hardware counters at four row alignments, same kernel | `e8_unitrace.sh` | `data/e8/` | Memory Alignment, appendix |
| E9 | Steady-state K sweep and M=K sweep | `e9_chunk_template.sh` (via `analysis/e9_plan.py`) | `data/e9/` | K-sweep and M=K figures, alignment table |
| E10 | Power limits with 1, 2, 6 and 12 busy stacks | `e10_allstacks.sh` | `data/e10/` | Power and Frequency, appendix |
| E11a | Which points of the original transformer sweeps ran in the boost | (analysis only) | `data/e11a/` | Motivated the steady-state reruns |
| attn | SDPA and Triton FA-2 throughput, kernel IDs, concurrency | `attn_smoke.sh`, `attn_full.sh`, `attn_conc.sh` | `data/attn/` | Fused Attention Results, attention figure |

E9 ran as five jobs of under an hour. `analysis/e9_plan.py <n>` writes the points that no earlier chunk measured and the payload
for chunk `n`, and `analysis/merge_e9.py` merges the chunks into the files in `data/e9/`.

## Data

- `data/e9/e9_fig2_points.csv` has one row per point of the K sweep (`m = n = 2048`, 23,967 values of `k`) with the median,
  10th and 90th percentile call time and throughput, the job chunk and node. `e9_fig2_basicGemmKSweep.csv` and
  `e9_fig3_basicGemmMKSweep.csv` use the `m,n,k,throughput` format of the original sizing CSVs.
- `gemm_bench.py` CSVs (`e3.csv`, `e5.csv`, `e6.csv` and others) have one row per shape with min, p10, median, p90 and max call
  time. Files named `*.joined.csv` add the replay's prediction for each row.
- `data/e4c/e4c.csv.iters.csv` has every call of the power-trace runs, and `e4c_smi_100ms.txt` the matching `xpu-smi` samples.
- `data/attn/attn_all.csv` has every attention measurement with its clock and power. `attn_summary.txt` is the readable summary.
- `*_summary.txt` files are the text outputs of the analysis scripts for each experiment.

Raw run directories (full logs, per-call timings of the other experiments, all counter groups) stayed on Aurora. The notes
refer to them as `runs/<name>_<date>/`.

## Differences from the benchmarks in `..`

`benchmark_mm` in `../utils.py` creates its operands on the host for every shape, which leaves the stack idle before the
timed calls, and it takes its statistic over the warm-up calls as well as the timed ones (the minimum in this branch, the
median in the copy used for the original Aurora sweeps). Both are fine on GPUs without a short power boost. On
Aurora, use `steady_sweep.py` or `gemm_bench.py` with a pre-heat for new measurements, and avoid running one process per stack
on all 12 stacks when you want single-stack numbers.

## Attribution

`replay/catalog.json` is derived from oneDNN's `kernel.db` (Apache License 2.0), and `replay/selector_sim.py` follows the
structure of oneDNN's selector and evaluator. `attn/tri_fa_kernels.py` is a verbatim copy of the Triton FlashAttention-2
forward kernel in ringX-attention, which derives from the Triton fused-attention tutorial. Its license is in
`attn/LICENSE.ringX`.
