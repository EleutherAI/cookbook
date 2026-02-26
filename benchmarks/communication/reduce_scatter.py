import torch
import sys, os, time

COMMS_BENCH_DIR = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(COMMS_BENCH_DIR)

from communication.utils import *
from communication.constants import *


def _reduce_scatter_op(dist, output, input, async_op=False):
    """Run a single reduce_scatter operation, returning the handle if async."""
    if hasattr(torch.distributed, "reduce_scatter_tensor"):
        return dist.reduce_scatter_tensor(output, input, async_op=async_op)
    elif hasattr(torch.distributed, "_reduce_scatter_base"):
        return dist._reduce_scatter_base(output, input, async_op=async_op)
    else:
        input_tensors = list(torch.chunk(input, dist.get_world_size()))
        return dist.reduce_scatter(output, input_tensors, async_op=async_op)


def timed_reduce_scatter(input, start_event, end_event, args, compute_tensors=None):
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist

    world_size = dist.get_world_size()
    # Create output tensor for reduce_scatter
    output = torch.empty(input.size(0) // world_size, dtype=input.dtype, device=input.device)

    sync_all()
    # Warmups, establish connections, etc.
    for i in range(args.warmups):
        if args.compute_overlap and compute_tensors is not None:
            handle = _reduce_scatter_op(dist, output, input, async_op=True)
            compute_kernel(compute_tensors, args.compute_iters)
            if handle is not None:
                handle.wait()
        else:
            _reduce_scatter_op(dist, output, input, async_op=args.async_op)
    sync_all()

    if args.compute_overlap and compute_tensors is not None:
        # Measure comm-only time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        sync_all()
        start.record()
        for _ in range(args.trials):
            _reduce_scatter_op(dist, output, input, async_op=False)
        end.record()
        sync_all()
        t_comm = start.elapsed_time(end) / 1000 / args.trials

        # Measure compute-only time
        t_compute = measure_compute_time(compute_tensors, args)

        # Measure overlapped time
        sync_all()
        start_event.record()
        for i in range(args.trials):
            handle = _reduce_scatter_op(dist, output, input, async_op=True)
            compute_kernel(compute_tensors, args.compute_iters)
            if handle is not None:
                handle.wait()
        end_event.record()
        sync_all()
        duration = start_event.elapsed_time(end_event) / 1000
        avg_duration = duration / args.trials
        overlap_pct = compute_overlap_pct(t_comm, t_compute, avg_duration)
    else:
        # time the actual comm op trials times and average it
        start_event.record()
        for i in range(args.trials):
            _reduce_scatter_op(dist, output, input, async_op=args.async_op)
        end_event.record()
        sync_all()
        duration = start_event.elapsed_time(end_event) / 1000
        avg_duration = duration / args.trials
        overlap_pct = None

    # maintain and clean performance data
    size = input.element_size() * input.nelement()
    n = dist.get_world_size()
    tput, busbw = get_bw('reduce_scatter', size, avg_duration, args)
    tput_str, busbw_str, duration_str = get_metric_strings(args, tput, busbw, avg_duration)
    desc = f'{input.nelement()}x{input.element_size()}'

    if not args.raw:
        size = convert_size(size)

    if overlap_pct is not None:
        print_rank_0(
            f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s} {overlap_pct:>11.1f}%")
    else:
        print_rank_0(f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}")


def run_reduce_scatter(local_rank, args):
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist

    # Prepare benchmark header
    print_header(args, 'reduce_scatter')

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    compute_tensors = create_compute_tensors(args, local_rank) if args.compute_overlap else None

    if args.scan:
        M_LIST = []
        for x in (2**p for p in range(1, args.maxsize)):
            M_LIST.append(x)

        sync_all()
        # loop over various tensor sizes
        for M in M_LIST:
            global_rank = dist.get_rank()
            try:
                # Ensure tensor size is divisible by world_size for reduce_scatter
                M = M - (M % world_size) if M % world_size != 0 else M
                mat = torch.ones(world_size, M,
                               dtype=getattr(torch, args.dtype)).cuda(local_rank)
                sync_all()
                input = ((mat.mul_(float(global_rank))).view(-1))
                del mat
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if dist.get_rank() == 0:
                        print('WARNING: Ran out of GPU memory. Exiting comm op.')
                    sync_all()
                    break
                else:
                    raise e
            sync_all()
            timed_reduce_scatter(input, start_event, end_event, args, compute_tensors)
    else:
        # Send the biggest message size our GPUs can fit. If you're facing OOM errors, reduce the mem_factor
        elements_per_gpu = max_numel(comm_op='reduce_scatter',
                                   dtype=getattr(torch, args.dtype),
                                   mem_factor=args.mem_factor * 2,
                                   local_rank=local_rank,
                                   args=args)
        # Ensure elements_per_gpu is divisible by world_size
        elements_per_gpu = elements_per_gpu - (elements_per_gpu % world_size) if elements_per_gpu % world_size != 0 else elements_per_gpu
        try:
            mat = torch.ones(elements_per_gpu, dtype=getattr(torch,
                                                           args.dtype)).cuda(local_rank)
            input = ((mat.mul_(float(global_rank))).view(-1))
        except RuntimeError as e:
            if 'out of memory' in str(e):
                if dist.get_rank() == 0:
                    print('WARNING: Ran out of GPU memory. Try to reduce the --mem-factor argument!')
                sync_all()
                return
            else:
                raise e
        sync_all()
        timed_reduce_scatter(input, start_event, end_event, args, compute_tensors)


if __name__ == "__main__":
    args = benchmark_parser().parse_args()
    rank = args.local_rank
    init_processes(local_rank=rank, args=args)
    run_reduce_scatter(local_rank=rank, args=args)