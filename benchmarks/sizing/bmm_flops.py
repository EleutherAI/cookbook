import time
import torch
import numpy as np
import sys
import argparse

from utils import benchmark_bmm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    b_group = parser.add_mutually_exclusive_group(required=True)
    b_group.add_argument("-b", nargs="+", type=int, help='The batched dimension of the BMM, enter any number of arguments')
    b_group.add_argument("--b_range", nargs='+', type=int, help="The batched dimension of the BMM, [start,stop,step]")

    m_group = parser.add_mutually_exclusive_group(required=True)
    m_group.add_argument("-m", nargs="+", type=int, help='The first dimension of the BMM, enter any number of arguments')
    m_group.add_argument("--m_range", nargs='+', type=int, help="The first dimension of the BMM, [start,stop,step]")

    n_group = parser.add_mutually_exclusive_group(required=True)
    n_group.add_argument("-n", nargs="*", type=int, help='The shared dimension of the BMM, enter any number of arguments')
    n_group.add_argument("--n_range", nargs='+', type=int, help="The shared dimension of the BMM, [start,stop,step]")

    k_group = parser.add_mutually_exclusive_group(required=True)
    k_group.add_argument("-k", nargs="*", type=int, help='The last dimension of the BMM, enter any number of arguments')
    k_group.add_argument("--k_range", nargs='+', type=int, help="The last dimension of the BMM, [start,stop,step]")

    parser.add_argument("--num_iterations", type=int, default=200, help='The number of iterations used to benchmark each BMM')
    parser.add_argument("--num_warmup_iterations", type=int, default=50, help='The number of warmup iterations')
    parser.add_argument("--cuda_device", type=int, default=0, help="The cuda device to run the benchmark on")
    parser.add_argument("--output_file", type=str, default="../results/bmm.out")
    args = parser.parse_args()

    b = args.b
    m = args.m
    n = args.n
    k = args.k

    if b is None:
        start,stop,step = args.b_range
        b = np.arange(start,stop,step)
    if m is None:
        start,stop,step = args.m_range
        m = np.arange(start,stop,step)
    if n is None:
        start,stop,step = args.n_range
        n = np.arange(start,stop,step)
    if k is None:
        start,stop,step = args.k_range
        k = np.arange(start,stop,step)
    
    # set cuda device
    torch.cuda.set_device(f"cuda:{args.cuda_device}")

    # loop through all sizes to benchmark
    with open(args.output_file, 'w') as sys.stdout:
        for B in b:
            for M in m:
                for N in n:
                    for K in k:
                        benchmark_bmm(B, M, N, K, "bmm", args.num_iterations, args.num_warmup_iterations)
                        print("-" * 80)
