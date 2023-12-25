import time
import torch
import sys


def benchmark_mm(m, n, k, num_iterations=100):
    A = torch.randn(m, n).half().to("cuda:0")
    B = torch.randn(n, k).half().to("cuda:0")
    C = torch.empty(m, k).half().to("cuda:0")
    num_warmup_iterations = 50
    for i in range(num_warmup_iterations + num_iterations):
        if i == num_warmup_iterations:
            start_time = time.time()
        with torch.no_grad():
            torch.mm(A, B, out=C)
        torch.cuda.synchronize()
    elapsed_time = (time.time() - start_time) / num_iterations
    print(f"Elapsed time for {m}x{n}x{k}: {elapsed_time:.3f}")
    print(f"Throughput (in TFLOP/s) for {m}x{n}x{k}: {(2 * m * n * k) / (elapsed_time * 10**12):.3f}")
    print("-" * 80)

def benchmark_mm_b(m, n, k, b=None, num_iterations=100):
    
    B = torch.randn(k, n).half().to("cuda:0")
    if b is None:
        A = torch.randn(m, n).half().to("cuda:0")
        b=1
        C = torch.empty(m, k).half().to("cuda:0")
    else:
        A = torch.randn(b,m,n).half().to("cuda:0")
        C = torch.empty(b,m, k).half().to("cuda:0")
    num_warmup_iterations = 50
    for i in range(num_warmup_iterations + num_iterations):
        if i == num_warmup_iterations:
            start_time = time.time()
        with torch.no_grad():
            torch.nn.functional.linear(A, B, out=C)
        torch.cuda.synchronize()
    elapsed_time = (time.time() - start_time) / num_iterations
    if b is None:
        print(f"Elapsed time for {m}x{n}x{k}: {elapsed_time:.3f}")
        print(f"Throughput (in TFLOP/s) for {m}x{n}x{k}: {(2 * m * n * k) / (elapsed_time * 10**12):.3f}")
    else:
        print(f"Elapsed time for {m}x{n}x{k}, b={b}: {elapsed_time:.4f}")
        print(f"Throughput (in TFLOP/s) for {m}x{n}x{k}, b={b}: "
          f"{(2 * b * m * n * k) / (elapsed_time * 10**12):.3f}") 
    print("-" * 80)

if __name__ == '__main__':
    torch.cuda.set_device("cuda:0")

    arch = "a100"

    # GEMM m sweep
    with open(f'../results/gemm_data/mm/m_sweep/{arch}.out', 'w') as sys.stdout:
        for log_size in range(5, 14):
            benchmark_mm(2**log_size, 4096, 2**log_size)

    # GEMM k sweep low
    with open(f'../results/gemm_data/mm/ksweep_low/{arch}.out', 'w') as sys.stdout:
        for k in range(64, 512, 2):
            benchmark_mm(27648, 4096, k)

    #  GEMM k sweep high
    with open(f'../results/gemm_data/mm/ksweep_high/{arch}.out', 'w') as sys.stdout:
        for k in range(1536, 6208, 64):
            benchmark_mm(2304, 4096, k)
