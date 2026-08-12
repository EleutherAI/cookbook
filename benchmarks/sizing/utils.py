import datetime
import platform
import sys
import shlex
import time
import torch
import numpy as np
from pathlib import Path
from torch.nn import LayerNorm
from megatron.model.fused_softmax import FusedScaleMaskSoftmax, SoftmaxFusionTypes
from megatron.model.transformer import ParallelSelfAttention, ParallelMLP, ParallelTransformerLayer, fused_attention
from megatron.model.transformer import bias_dropout_add_fused_train
from megatron.model.activations import bias_gelu_impl
from megatron.model.gpt2_model import gpt2_attention_mask_func as attention_mask_func
from megatron.model.word_embeddings import Embedding
from torch.nn.functional import scaled_dot_product_attention
import gc

def print_benchmark_header(device,notes="None"):
    
    print(f"""
Benchmark started on {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}

** Command line:
{sys.executable} {" ".join(map(shlex.quote, sys.argv))}

** Platform:
{" ".join(platform.uname())}
{get_device_properties(device)}

** Critical component versions:
torch={torch.__version__}, cuda={torch.version.cuda if device.type == device
else torch.version.xpu}, nccl={torch.cuda.nccl.version() if device.type ==
device else "ccl version unavailable through API"}

** Additional notes: 
{notes}

{"-" * 80}

""")

class Tee(object):
    def __init__(self, filename, verbose):
        Path(filename).resolve().parent.mkdir(parents=True, exist_ok=True)
        self.file = open(filename, "a")
        self.verbose = verbose
        if self.verbose:
            self.stdout = sys.stdout

    def write(self, message):
        self.file.write(message)
        if self.verbose:
            self.stdout.write(message)

    def flush(self):
        self.file.flush()
        if self.verbose:
            self.stdout.flush()


def display(shape):
    return "x".join([str(dim) for dim in shape])

def get_device():
    if torch.cuda.is_available():
        return torch.device(device)
    elif torch.xpu.is_available():
        return torch.device("xpu")

def get_device_properties(device):
    if device == None:
        return None
    if device.type == device:
        return torch.cuda.get_device_properties(device)
    if device.type == "xpu":
        return torch.xpu.get_device_properties(device)


def set_device(device_num : int = 0):
    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{device_num}")
    elif torch.xpu.is_available():
        torch.xpu.set_device(f"xpu:{device_num}")

def device_synchronize(device):
    if device.type == device:
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)

def get_device_timing_event(device):
    if device.type == device:
        return torch.cuda.Event(enable_timing=True)
    elif device.type == "xpu":
        return torch.xpu.Event(enable_timing=True)

def start_prof(args, bench_idx, prof):
    should_start = (
        args.profile
        and  prof is None
        and args.profile_start_idx is not None
        and bench_idx == args.profile_start_idx
    )

    if should_start:
        print(f"Starting profiler at index: {bench_idx}")

        activities = [torch.profiler.ProfilerActivity.CPU]

        if hasattr(torch.profiler.ProfilerActivity, "CUDA"):
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)

        if hasattr(torch.profiler.ProfilerActivity, "XPU"):
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                activities.append(torch.profiler.ProfilerActivity.XPU)
        prof = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

        prof.__enter__()

    return prof

def stop_prof(args, bench_idx, prof):
    should_stop = (
        prof is not None
        and args.profile_stop_idx is not None
        and bench_idx == args.profile_stop_idx
    )

    if should_stop:
        print(f"Stopping profiler at index: {bench_idx}")

        prof.__exit__(None, None, None)

        print(
            prof.key_averages().table(
                sort_by="self_cuda_time_total",
                row_limit=50,
            )
        )

        prof.export_chrome_trace(args.profile_output)

        prof = None
    
    return prof

# Benchmark of a basic GEMM
def benchmark_mm(m, n, k, num_iterations, num_warmup_iterations):
    device = get_device()
    start = get_device_timing_event(device)
    end = get_device_timing_event(device)

    A = torch.randn(m, n).bfloat16().to(device)
    B = torch.randn(n, k).bfloat16().to(device)
    C = torch.empty(m, k).bfloat16().to(device)
    times = np.zeros(num_iterations+num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            torch.mm(A, B, out=C)
            end.record()
        device_synchronize(device)
        times[i] = start.elapsed_time(end)
    gc.collect()
    torch.cuda.empty_cache()
    elapsed_time = np.amin(times)/1000 
    print(f"Elapsed time for {m}x{n}x{k}: {elapsed_time:.3f}")
    print(f"Throughput (in TFLOP/s) for {m}x{n}x{k}: {(2 * m * n * k) / (elapsed_time * 10**12):.3f}")
    print("-" * 80)
    return elapsed_time

# Benchmark of a GEMM with a single batched operator
def benchmark_mm_b(m, n, k, label, b, num_iterations,num_warmup_iterations):
    device = get_device()
    start = get_device_timing_event(device)
    end = get_device_timing_event(device)

    B = torch.randn((k, n)).bfloat16().to(device)
    if b is None:
        A = torch.randn((m, n)).bfloat16().to(device)
        C = torch.empty((m, k)).bfloat16().to(device)
        b = 1
    else:
        A = torch.randn((b, m, n)).bfloat16().to(device)
        C = torch.empty((b, m, k)).bfloat16().to(device)
    times = np.zeros(num_iterations+num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            torch.nn.functional.linear(A, B, out=C)
            end.record()
        device_synchronize(device)
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amin(times)/1000 
    print(f"Elapsed time for {label} ({m}x{n}x{k}, b={b}): {elapsed_time :.4f}")
    print(f"Throughput (in TFLOP/s) for {label} ({m}x{n}x{k}, b={b}): "
          f"{(2 * b * m * n * k) / (elapsed_time * 10**12):.3f}")
    return elapsed_time

def benchmark_bmm(b, m, n, k, label,num_iterations, num_warmup_iterations):
    device = get_device()
    start = get_device_timing_event(device)
    end = get_device_timing_event(device)
    A = torch.randn((b, m, n)).bfloat16().to(device)
    B = torch.randn((b, n, k)).bfloat16().to(device)
    C = torch.empty((b, m, k)).bfloat16().to(device)
    times = np.zeros(num_iterations+num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            torch.bmm(A, B, out=C)
            end.record()
        device_synchronize(device)
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amin(times)/1000 
    print(f"Elapsed time for {label} ({b}x{m}x{n}x{k}): {elapsed_time :.4f}")
    print(f"Throughput (in TFLOP/s) for {label} ({b}x{m}x{n}x{k}): "
          f"{(2 * b * m * n * k) / (elapsed_time * 10**12):.3f}")
    return elapsed_time

def benchmark_dropout(A_dim, label, num_iterations, num_warmup_iterations):
    device = get_device()
    start = get_device_timing_event(device)
    end = get_device_timing_event(device)
    A = torch.randn(A_dim).bfloat16().to(device)
    dropout = torch.nn.Dropout(0.5).to(device)

    times = np.zeros(num_iterations+num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            dropout(A)
            end.record()
        device_synchronize(device)
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amin(times)/1000 
    print(f"Elapsed time for {label} ({display(A_dim)}): {elapsed_time :.4f}")
    return elapsed_time

def benchmark_softmax(scores_shape, seq_length, label, num_iterations,num_warmup_iterations):
    device = get_device()
    start = get_device_timing_event(device)
    end = get_device_timing_event(device)
    scores = torch.randn(scores_shape).bfloat16().to(device)
    attention_mask = torch.tril(torch.ones(
        (1, seq_length, seq_length), device=device)).view(
        1, 1, seq_length, seq_length)
    attention_mask = attention_mask < 0.5
    softmax = FusedScaleMaskSoftmax(
        True, False,
        SoftmaxFusionTypes.none, #attentionmasktype.padding=1,True
        attention_mask_func, True, 1)
    times = np.zeros(num_iterations+num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            softmax(scores, attention_mask)
            end.record()
        device_synchronize(device)
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amin(times)/1000 
    print(f"Elapsed time for {label} ({display(scores_shape)}): {elapsed_time :.4f}")
    return elapsed_time


def benchmark_fused_gelu(A_dim, b_dim, label, num_iterations, num_warmup_iterations):
    device = get_device()
    start = get_device_timing_event(device)
    end = get_device_timing_event(device)
    A = torch.randn(A_dim).bfloat16().to(device)
    b = torch.randn(b_dim).bfloat16().to(device)
    times = np.zeros(num_iterations+num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            bias_gelu_impl(A, b)
            end.record()
        device_synchronize(device)
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amin(times)/1000 
    print(f"Elapsed time for {label} ({display(A_dim)}): {elapsed_time :.4f}")
    return elapsed_time

def benchmark_layer_norm(A_dim, normalized_shape, label, num_iterations, num_warmup_iterations):
    device = get_device()
    start = get_device_timing_event(device)
    end = get_device_timing_event(device)
    A = torch.randn(A_dim).bfloat16().to(device)
    layer_norm = LayerNorm(normalized_shape).bfloat16().to(device)
    times = np.zeros(num_iterations+num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            layer_norm(A)
            end.record()
        device_synchronize(device)
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amin(times)/1000 
    print(f"Elapsed time for {label} ({display(A_dim)}): {elapsed_time :.4f}")
    return elapsed_time

def benchmark_add_bias_dropout(shape, label, num_iterations, num_warmup_iterations):
    device = get_device()
    start = get_device_timing_event(device)
    end = get_device_timing_event(device)
    A = torch.randn(shape).bfloat16().to(device)
    bias = torch.randn(shape).bfloat16().to(device)
    residue = torch.randn(shape).bfloat16().to(device)
    times = np.zeros(num_iterations+num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            bias_dropout_add_fused_train(A, bias, residue, 0.0)
            end.record()
        device_synchronize(device)
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amin(times)/1000 
    print(f"Elapsed time for {label} ({display(shape)}): {elapsed_time :.4f}")
    return elapsed_time

def benchmark_flash(shape, label, num_iterations, num_warmup_iterations):
    device = get_device()
    start = get_device_timing_event(device)
    end = get_device_timing_event(device)
    q = torch.randn(shape).bfloat16().to(device)
    k = torch.randn(shape).bfloat16().to(device)
    v = torch.randn(shape).bfloat16().to(device)
    times = np.zeros(num_iterations+num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            fused_attention(q,k,v)
            end.record()
        device_synchronize(device)
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amin(times)/1000 
    # 4bhs^2
    throughput = (4 * shape[0] * shape[1]**2 * shape[2] *shape[3]) / (elapsed_time * 10**12)
    print(f"Throughput (in TFLOP/s) for {label} ({display(shape)}): {throughput:.3f}")
    print(f"Elapsed time for {label} ({display(shape)}): {elapsed_time :.4f}")
    return elapsed_time

def benchmark_sdpa(shape, label, num_iterations, num_warmup_iterations):
    device = get_device()
    start = get_device_timing_event(device)
    end = get_device_timing_event(device)
    q = torch.randn(shape).bfloat16().to(device)
    k = torch.randn(shape).bfloat16().to(device)
    v = torch.randn(shape).bfloat16().to(device)
    times = np.zeros(num_iterations+num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            scaled_dot_product_attention(q,k,v)
            end.record()
        device_synchronize(device)
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amin(times)/1000 
    # 4bhs^2
    throughput = (4 * shape[0] * shape[2]**2 * shape[1] *shape[3]) / (elapsed_time * 10**12)
    print(f"Throughput (in TFLOP/s) for {label} ({display(shape)}): {throughput:.3f}")
    print(f"Elapsed time for {label} ({display(shape)}): {elapsed_time :.4f}")
    return elapsed_time


