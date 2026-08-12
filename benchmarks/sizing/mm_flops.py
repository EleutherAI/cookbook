import time
import torch
import sys
import numpy as np
import argparse
import os
import torch.profiler
from contextlib import nullcontext

from utils import Tee, benchmark_mm, print_benchmark_header, set_device, get_device, start_prof, stop_prof

file_dir = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    m_group = parser.add_mutually_exclusive_group(required=True)
    m_group.add_argument("-m", nargs="+", type=int, help='The first dimension of the GEMM, enter any number of arguments')
    m_group.add_argument("--m_range", nargs='+', type=int, help="The first dimension of the GEMM, [start,stop,step]")

    n_group = parser.add_mutually_exclusive_group(required=True)
    n_group.add_argument("-n", nargs="*", type=int, help='The shared dimension of the GEMM, enter any number of arguments')
    n_group.add_argument("--n_range", nargs='+', type=int, help="The shared dimension of the GEMM, [start,stop,step]")

    k_group = parser.add_mutually_exclusive_group(required=True)
    k_group.add_argument("-k", nargs="*", type=int, help='The last dimension of the GEMM, enter any number of arguments')
    k_group.add_argument("--k_range", nargs='+', type=int, help="The last dimension of the GEMM, [start,stop,step]")

    parser.add_argument("--num_iterations", type=int, default=200, help='The number of iterations used to benchmark each GEMM')
    parser.add_argument("--num_warmup_iterations", type=int, default=50, help='The number of warmup iterations')
    parser.add_argument("--device", type=int, default=0, help="The device to run the benchmark on")
    parser.add_argument("--output_file", type=str, default=f"{file_dir}/results/mm.out")
    parser.add_argument("--notes", type=str, default="", help="benchmark-specific notes to add to the output_file's header")
    parser.add_argument("--verbose", default=True, action=argparse.BooleanOptionalAction, help='log to stdout besides output_file?')
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorch profiler"
    )

    parser.add_argument("--profile_start_idx", type=int, default=None)
    parser.add_argument("--profile_stop_idx", type=int, default=None)
    parser.add_argument(
        "--profile_output",
        type=str,
        default="profile_trace.json"
    )
    args = parser.parse_args()

    m = args.m
    n = args.n
    k = args.k

    if m is None:
        start,stop,step = args.m_range
        m = np.arange(start,stop,step)
    if n is None:
        start,stop,step = args.n_range
        n = np.arange(start,stop,step)
    if k is None:
        start,stop,step = args.k_range
        k = np.arange(start,stop,step)
    
    # set device
    set_device(args.device)
    device = get_device()

    sys.stdout = Tee(args.output_file, args.verbose)
    print_benchmark_header(device,args.notes)

    prof = None
    bench_idx = 0

    for M in m:
        for N in n:
            for K in k:

                prof = start_prof(args, bench_idx, prof)    

                benchmark_mm(
                    M,
                    N,
                    K,
                    args.num_iterations,
                    args.num_warmup_iterations,
                )
                bench_idx += 1

                prof = stop_prof(args, bench_idx, prof)


