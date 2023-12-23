# Communication Benchmarks

The intent of these benchmarks is to measure communication latency/bandwidth of DeepSpeed and/or pytorch distributed communication operations at the Python layer. These benchmarks are complementary to C-level comms benchmarks like [OSU Micro-Benchmarks](https://mvapich.cse.ohio-state.edu/benchmarks/) and [NCCL Tests](https://github.com/NVIDIA/nccl-tests) in that users can:
- Easily debug which layer of the communication software stack hangs or performance degradations originate from.
- Measure the expected communication performance of either DeepSpeed comms or pure PyTorch distributed

To run benchmarks, there are two options:

1. Run a single communication operation:

For example, run with a single large message size (calculated to barely fit within GPU mem):
<pre>
mpirun -np 16 --hostfile ${HOSTFILE} -x LD_LIBRARY_PATH -x PATH -x LD_PRELOAD python all_reduce.py
</pre>

Scan across message sizes:
<pre>
mpirun -np 16 --hostfile ${HOSTFILE} -x LD_LIBRARY_PATH -x PATH -x LD_PRELOAD python all_reduce.py --scan
</pre>

Benchmark pure PyTorch distributed comms (without importing or using MCR-DL) by launching with MPI
<pre>
mpirun -np 16 --hostfile ${HOSTFILE} -x LD_LIBRARY_PATH -x PATH -x LD_PRELOAD python all_reduce.py --scan --dist="torch"
</pre>

or Slurm
<pre>
srun -n 16 python all_reduce.py --scan --dist="torch"
</pre>


2. Run all available communication benchmarks:

<pre>
mpirun -np 16 --hostfile ${HOSTFILE} -x LD_LIBRARY_PATH -x PATH -x LD_PRELOAD python run_all.py
</pre>

Like the individual benchmarks, `run_all.py` supports scanning arguments for the max message size, bandwidth-unit, etc. Simply pass the desired arguments to `run_all.py` and they'll be propagated to each comm op.

Finally, users can choose specific communication operations to run in `run_all.py` by passing them as arguments (all operations are run by default). For example:

<pre>
mpirun -np 16 --hostfile ${HOSTFILE} -x LD_LIBRARY_PATH -x PATH -x LD_PRELOAD python run_all.py --scan --all-reduce --all-to-all --broadcast
</pre>


# Adding Communication Benchmarks

To add new communication benchmarks, follow this general procedure:

1. Copy a similar benchmark file (e.g. to add `reduce_scatter`, copy `all_reduce.py` as a template)
2. Add a new bandwidth formula in `utils.get_bandwidth`, a new maximum tensor element formula in `utils.max_numel`, and a new arg in `utils.benchmark_parser`
3. Replace comm op calls in new file with find-replace
4. Find a good default `mem_factor` for use in `run_<collective>_single()` function
5. Add new comm op to `run_all.py`
