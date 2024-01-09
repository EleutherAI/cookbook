import time
import torch
import numpy as np
import sys

def benchmark_bmm(b, m, n, k, num_iterations=200):
    A = torch.randn((b, m, n)).half().to("cuda:0")
    B = torch.randn((b, n, k)).half().to("cuda:0")
    C = torch.empty((b, m, k)).half().to("cuda:0")
    num_warmup_iterations=50
    times = np.zeros(num_iterations+num_warmup_iterations)
    start_time = time.time()
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            torch.bmm(A, B, out=C)
        torch.cuda.synchronize()
        times[i] = time.time()

    #elapsed_time = (time.time() - start_time) / num_iterations
    times -= start_time
    times = np.diff(times)
    times = times[50:]
    elapsed_time = np.amax(times)
    print(f"Elapsed time for {b}x{m}x{n}x{k}: {elapsed_time:.3f}")
    print(f"Throughput (in TFLOP/s) for {b}x{m}x{n}x{k}: {(2 * b * m * n * k) / (elapsed_time * 10**12):.3f}")
    flops = (2 * b * m * n * k) / (elapsed_time * 10**12)
    print("-" * 80)
    return flops

if __name__ == '__main__':
    torch.cuda.set_device("cuda:0")

    arch = "a100"

    # Try to determine the effect of b on throughput with square individual MMs.
    with open(f'../results/gemm_data/bmm/bsweep/{arch}.out', 'w') as sys.stdout:
        for log_b in range(7):
            b = 2**log_b
            benchmark_bmm(b, m=1024, n=1024, k=1024)
            benchmark_bmm(b, m=2048, n=2048, k=2048)
            benchmark_bmm(b, m=4096, n=4096, k=4096)
            benchmark_bmm(b, m=8192, n=8192, k=8192)
    
    # Try to determine the effect of b and outer_dim on throughput with non-square
    # individual MMs.
    with open(f'../results/gemm_data/bmm/msweep/{arch}.out', 'w') as sys.stdout:
        for log_b in range(7):
            b = 2**log_b
            for log_outer_dim in range(5, 14):
                outer_dim = 2**log_outer_dim
                benchmark_bmm(b, m=outer_dim, n=4096, k=outer_dim)
