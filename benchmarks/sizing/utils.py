import sys
import torch
import numpy as np
from pathlib import Path
from megatron.model import LayerNorm
from megatron.model.fused_softmax import FusedScaleMaskSoftmax, SoftmaxFusionTypes
from megatron.model.transformer import ParallelSelfAttention, ParallelMLP, ParallelTransformerLayer
from megatron.model.transformer import bias_dropout_add_fused_train
from megatron.model.activations import bias_gelu_impl
from megatron.model.gpt2_model import gpt2_attention_mask_func as attention_mask_func
from megatron.model.word_embeddings import Embedding

class Tee(object):
    def __init__(self, filename, verbose):
        Path(filename).resolve().parent.mkdir(parents=True, exist_ok=True)
        self.file = open(filename, "w")
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

# Benchmark of a basic GEMM
def benchmark_mm(m, n, k, num_iterations, num_warmup_iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    A = torch.randn(m, n).half().to("cuda")
    B = torch.randn(k, n).half().to("cuda")
    #B = torch.randn(n, k).half().to("cuda")
    C = torch.empty(m, k).half().to("cuda")
    times = np.zeros(num_iterations+num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            torch.nn.functional.linear(A, B, out=C)
            #torch.mm(A, B, out=C)
            end.record()
        torch.cuda.synchronize()
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amax(times)/1000 
    print(f"Elapsed time for {m}x{n}x{k}: {elapsed_time:.3f}")
    print(f"Throughput (in TFLOP/s) for {m}x{n}x{k}: {(2 * m * n * k) / (elapsed_time * 10**12):.3f}")
    print("-" * 80)
    return elapsed_time

# Benchmark of a GEMM with a single batched operator
def benchmark_mm_b(m, n, k, label, b, num_iterations,num_warmup_iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    B = torch.randn((k, n)).half().to("cuda")
    if b is None:
        A = torch.randn((m, n)).half().to("cuda")
        C = torch.empty((m, k)).half().to("cuda")
        b = 1
    else:
        A = torch.randn((b, m, n)).half().to("cuda")
        C = torch.empty((b, m, k)).half().to("cuda")
    times = np.zeros(num_iterations+num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            torch.nn.functional.linear(A, B, out=C)
            end.record()
        torch.cuda.synchronize()
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amax(times)/1000 
    print(f"Elapsed time for {label} ({m}x{n}x{k}, b={b}): {elapsed_time :.4f}")
    print(f"Throughput (in TFLOP/s) for {label} ({m}x{n}x{k}, b={b}): "
          f"{(2 * b * m * n * k) / (elapsed_time * 10**12):.3f}")
    return elapsed_time

def benchmark_bmm(b, m, n, k, label,num_iterations, num_warmup_iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    A = torch.randn((b, m, n)).half().to("cuda")
    B = torch.randn((b, n, k)).half().to("cuda")
    C = torch.empty((b, m, k)).half().to("cuda")
    times = np.zeros(num_iterations+num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            torch.bmm(A, B, out=C)
            end.record()
        torch.cuda.synchronize()
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amax(times)/1000 
    print(f"Elapsed time for {label} ({b}x{m}x{n}x{k}): {elapsed_time :.4f}")
    print(f"Throughput (in TFLOP/s) for {label} ({b}x{m}x{n}x{k}): "
          f"{(2 * b * m * n * k) / (elapsed_time * 10**12):.3f}")
    return elapsed_time

def benchmark_dropout(A_dim, label, num_iterations, num_warmup_iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    A = torch.randn(A_dim).half().to("cuda")
    dropout = torch.nn.Dropout(0.5).to("cuda")

    times = np.zeros(num_iterations+num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            dropout(A)
            end.record()
        torch.cuda.synchronize()
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amax(times)/1000 
    print(f"Elapsed time for {label} ({display(A_dim)}): {elapsed_time :.4f}")
    return elapsed_time

def benchmark_softmax(scores_shape, seq_length, label, num_iterations,num_warmup_iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    scores = torch.randn(scores_shape).half().to("cuda")
    attention_mask = torch.tril(torch.ones(
        (1, seq_length, seq_length), device="cuda")).view(
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
        torch.cuda.synchronize()
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amax(times)/1000 
    print(f"Elapsed time for {label} ({display(scores_shape)}): {elapsed_time :.4f}")
    return elapsed_time

def benchmark_fused_gelu(A_dim, b_dim, label, num_iterations, num_warmup_iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    A = torch.randn(A_dim).half().to("cuda")
    b = torch.randn(b_dim).half().to("cuda")
    times = np.zeros(num_iterations+num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            bias_gelu_impl(A, b)
            end.record()
        torch.cuda.synchronize()
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amax(times)/1000 
    print(f"Elapsed time for {label} ({display(A_dim)}): {elapsed_time :.4f}")
    return elapsed_time

def benchmark_layer_norm(A_dim, normalized_shape, label, num_iterations, num_warmup_iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    A = torch.randn(A_dim).half().to("cuda")
    layer_norm = LayerNorm(normalized_shape).half().to("cuda")
    times = np.zeros(num_iterations+num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            layer_norm(A)
            end.record()
        torch.cuda.synchronize()
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amax(times)/1000 
    print(f"Elapsed time for {label} ({display(A_dim)}): {elapsed_time :.4f}")
    return elapsed_time

def benchmark_add_bias_dropout(shape, label, num_iterations, num_warmup_iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    A = torch.randn(shape).half().to("cuda")
    bias = torch.randn(shape).half().to("cuda")
    residue = torch.randn(shape).half().to("cuda")
    times = np.zeros(num_iterations+num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            bias_dropout_add_fused_train(A, bias, residue, 0.0)
            end.record()
        torch.cuda.synchronize()
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amax(times)/1000 
    print(f"Elapsed time for {label} ({display(shape)}): {elapsed_time :.4f}")
    return elapsed_time
