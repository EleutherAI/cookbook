import time
import torch
import sys
import numpy as np
import argparse
import os

from utils import Tee, benchmark_mm, print_benchmark_header, set_device, get_device, benchmark_mm_b

file_dir = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':

    # Figure 3. basicGemmMSweep.out
    #for log_size in range(5, 14):
    #    benchmark_mm(2**log_size, 4096, 2**log_size)

    # Figure 7. basicGemmKSweep.out
    #for k in range(64, 2**15, 64):
    #    benchmark_mm(2048, 2048, k)

    # Figure 8. basicGemmLargeKSweep.out
    #for k in range(1536, 6208, 64):
    #    benchmark_mm(2304, 4096, k)

    # m from 1024 to 10000.
    #for m in range(64, 2**15, 64):
    #    benchmark_mm(m, 2048, 2048)

    #n from 64 to 512
    #for n in range(64, 2**15, 64):
    #    benchmark_mm(2048,n,2048)
    

    #for nk in range( 64, 2**15, 64):
    #    benchmark_mm(2048, 4*nk, nk)


    #for mn in range(64, 4096, 8):
    #    benchmark_mm(mn,2048,mn)

    #batch vs concat
    for n in range(64, 524288, 64):
        benchmark_mm_b(2048,n,2048, b=4, num_iterations=100,num_warmup_iterations=50,label="")
        benchmark_mm_b(1024,n,2048, b=8, num_iterations=100,num_warmup_iterations=50,label="")
        benchmark_mm_b(512,n,2048, b=16, num_iterations=100,num_warmup_iterations=50,label="")
        benchmark_mm_b(256,n,2048, b=32, num_iterations=100,num_warmup_iterations=50,label="")
        benchmark_mm(8192, n, 2048, num_iterations=100, num_warmup_iterations=50)
    
    #profile linear projection
    #benchmark_mm_b(4,13056,13056,b=2048)

    #sweep nk
    #for logB in range(4,6):
    #    B = 2**logB
    #    for n in range(64, 2**15, 64):
    #        benchmark_mm_b(2048, n, 2048, b=B)

    #sweep nk in area of low speed
    #for hidden_size in range(22976,25024+64,64):
    #    benchmark_mm_b(4, hidden_size, hidden_size, b=2048)

    #profile separate arbitrary region
    
    #for hidden_size in range( 64, 2**15, 64):
    #    benchmark_mm_b(4, 3*hidden_size, hidden_size, b=2048)

    #h to 4h drop
    #for h in range(128,2**15,128): 
    #    benchmark_mm_b(2048,h, 3*h, b=4)

    #for h in range(128,2**15,128): 
    #    benchmark_mm_b(4,h, 3*h, b=2048)

    #for h in range(128,2**15,128): 
    #    benchmark_mm_b(4*2048,h, 3*h)
