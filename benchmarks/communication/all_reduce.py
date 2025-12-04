import torch
import sys, os, time

COMMS_BENCH_DIR = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(COMMS_BENCH_DIR)

from communication.utils import *
from communication.constants import *


def timed_all_reduce(input, start_event, end_event, args):
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist

    sync_all()
    # Warmups, establish connections, etc.
    for i in range(args.warmups):
        dist.all_reduce(input, async_op=args.async_op)
    sync_all()

    # time the actual comm op trials times and average it
    start_event.record()
    for i in range(args.trials):
        dist.all_reduce(input, async_op=args.async_op)
    end_event.record()
    sync_all()
    duration = start_event.elapsed_time(end_event) / 1000

    # maintain and clean performance data
    avg_duration = duration / args.trials
    size = input.element_size() * input.nelement()
    n = dist.get_world_size()
    tput, busbw = get_bw('all_reduce', size, avg_duration, args)
    tput_str, busbw_str, duration_str = get_metric_strings(args, tput, busbw, avg_duration)
    desc = f'{input.nelement()}x{input.element_size()}'

    if not args.raw:
        size = convert_size(size)

    print_rank_0(f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}")


def create_input_tensor(world_size, num_elements, global_rank, local_rank, args):
    """
    Create input tensor for all_reduce benchmark/validation.
    
    Each rank's tensor is filled with its rank value, so after all_reduce SUM,
    each element should equal sum(0..n-1) = n*(n-1)/2.
    
    Args:
        world_size: Total number of ranks
        num_elements: Number of elements per rank
        global_rank: This rank's global rank
        local_rank: This rank's local rank
        args: Benchmark arguments
        
    Returns:
        Input tensor on GPU, or None if OOM
    """
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist

    try:
        mat = torch.ones(num_elements, dtype=getattr(torch, args.dtype)).cuda(local_rank)
        input = mat.mul_(float(global_rank))
        return input
    except RuntimeError as e:
        if 'out of memory' in str(e):
            if dist.get_rank() == 0:
                print('WARNING: Ran out of GPU memory.')
            sync_all()
            return None
        else:
            raise e


def run_all_reduce(local_rank, args):
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist

    # Prepare benchmark header unless validating
    if not args.validate:
        print_header(args, 'all_reduce')
    else:
        print_rank_0("Running Allreduce validation")

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    if args.single:
        sync_all()
        num_elements = world_size * (2 ** args.maxsize)
        input = create_input_tensor(world_size, num_elements, global_rank, local_rank, args)
        if input is None:
            if dist.get_rank() == 0:
                print('Exiting comm op.')
            return
        sync_all()
        
        if args.validate:
            run_validation(input, args)
        else:
            timed_all_reduce(input, start_event, end_event, args)

    elif args.scan:
        M_LIST = [2**p for p in range(1, args.maxsize)]

        sync_all()
        # loop over various tensor sizes
        for M in M_LIST:
            num_elements = world_size * M
            input = create_input_tensor(world_size, num_elements, global_rank, local_rank, args)
            if input is None:
                break
            sync_all()
            
            if args.validate:
                run_validation(input, args)
            else:
                timed_all_reduce(input, start_event, end_event, args)
            
            # Clean up for next iteration
            del input
            torch.cuda.empty_cache()
    else:
        # Send the biggest message size our GPUs can fit. If you're facing OOM errors, reduce the mem_factor
        # Don't need output tensor, so we double mem_factor
        elements_per_gpu = max_numel(comm_op='all_reduce',
                                     dtype=getattr(torch, args.dtype),
                                     mem_factor=args.mem_factor * 2,
                                     local_rank=local_rank,
                                     args=args)
        
        input = create_input_tensor(world_size, elements_per_gpu, global_rank, local_rank, args)
        if input is None:
            if dist.get_rank() == 0:
                print('Try to reduce the --mem-factor argument!')
            return
        sync_all()
        
        if args.validate:
            run_validation(input, args)
        else:
            timed_all_reduce(input, start_event, end_event, args)


if __name__ == "__main__":
    args = benchmark_parser().parse_args()
    rank = args.local_rank
    init_processes(local_rank=rank, args=args)
    run_all_reduce(local_rank=rank, args=args)