# Transformer Sizing Guidelines

The intent of these benchmarks is to measure the throughput of Generalized Matrix Multiplications (GEMMs) and Batched Matrix Multiplications (BMM) found in transformer models on modern GPU architectures. With these benchmarks, users can easily study:
- The performance characteristics of GEMMs and BMMs on their GPU architecture.
- How these GEMMs and BMMs form transformer layers.

## Dependencies

First, install the required packages:
```
pip install -r requirements.txt
```


There are three scripts within `benchmarks/sizing` that can be run:

## GEMM Benchmarks
`mm_flops.py` measures throughput of GEMMs of shape $(m, n) \times (n, k)$.
```
Example for mm_flops.py: python mm_flops.py -m 1024 -k 1024 -n 1024 2048
Example for mm_flops.py with range option: python mm_flops.py -m 1024 -k 1024 --n_range 1024 2048 256
usage: mm_flops.py [-h] (-m M [M ...] | --m_range M_RANGE [M_RANGE ...]) (-n [N ...] | --n_range N_RANGE [N_RANGE ...])(-k [K ...] | --k_range K_RANGE [K_RANGE ...]) [--num_iterations NUM_ITERATIONS]
[--num_warmup_iterations NUM_WARMUP_ITERATIONS] [--cuda_device CUDA_DEVICE] [--output_file OUTPUT_FILE]

options:
  -h, --help            show this help message and exit
  -m M [M ...]          The first dimension of the GEMM, enter any number of arguments
  --m_range M_RANGE [M_RANGE ...]
                        The first dimension of the GEMM, [start,stop,step]
  -n [N ...]            The shared dimension of the GEMM, enter any number of arguments
  --n_range N_RANGE [N_RANGE ...]
                        The shared dimension of the GEMM, [start,stop,step]
  -k [K ...]            The last dimension of the GEMM, enter any number of arguments
  --k_range K_RANGE [K_RANGE ...]
                        The last dimension of the GEMM, [start,stop,step]
  --num_iterations NUM_ITERATIONS
                        The number of iterations used to benchmark each GEMM
  --num_warmup_iterations NUM_WARMUP_ITERATIONS
                        The number of warmup iterations
  --cuda_device CUDA_DEVICE
                        The cuda device to run the benchmark on
  --output_file OUTPUT_FILE
  --verbose, --no-verbose
                        log to stdout besides output_file? (default: True)
```

## BMM Benchmarks
`bmm_flops.py` measures throughput of batched matrix multiplications $(b,m,n)\times (b,n,k)$.
```
Example for bmm_flops.py: python bmm_flops.py -m 1024 -k 1024 -n 1024 2048 -b 128
usage: bmm_flops.py [-h] (-b B [B ...] | --b_range B_RANGE [B_RANGE ...]) (-m M [M ...] | --m_range M_RANGE [M_RANGE ...])(-n [N ...] | --n_range N_RANGE [N_RANGE ...]) (-k [K ...] | --k_range K_RANGE [K_RANGE ...])
[--num_iterations NUM_ITERATIONS] [--num_warmup_iterations NUM_WARMUP_ITERATIONS] [--cuda_device CUDA_DEVICE][--output_file OUTPUT_FILE]

options:
  -h, --help            show this help message and exit
  -b B [B ...]          The batched dimension of the BMM, enter any number of arguments
  --b_range B_RANGE [B_RANGE ...]
                        The batched dimension of the BMM, [start,stop,step]
  -m M [M ...]          The first dimension of the BMM, enter any number of arguments
  --m_range M_RANGE [M_RANGE ...]
                        The first dimension of the BMM, [start,stop,step]
  -n [N ...]            The shared dimension of the BMM, enter any number of arguments
  --n_range N_RANGE [N_RANGE ...]
                        The shared dimension of the BMM, [start,stop,step]
  -k [K ...]            The last dimension of the BMM, enter any number of arguments
  --k_range K_RANGE [K_RANGE ...]
                        The last dimension of the BMM, [start,stop,step]
  --num_iterations NUM_ITERATIONS
                        The number of iterations used to benchmark each BMM
  --num_warmup_iterations NUM_WARMUP_ITERATIONS
                        The number of warmup iterations
  --cuda_device CUDA_DEVICE
                        The cuda device to run the benchmark on
  --output_file OUTPUT_FILE
  --verbose, --no-verbose
                        log to stdout besides output_file? (default: True)
```

Note that `bmm` with `b=1` performs about the same as `mm` starting from largish dimensions [see](https://gist.github.com/malfet/6a17156d7f5663b8b12054a1beff3fe1).

## Transformer Layer Benchmarks
`transformer_flops.py` measures throughput of a transformer layer or of each block of a transformer layer.
```
Example for transformer_flops.py: python transformer_flops.py --hidden_size 4096 --num_attention_heads 16 --microbatch_size 4 --seq_length 2048 --vocab_size 51200 --global_batch_size 256 --tensor_mp_size 1 --num_iterations 10 --num_warmup_iterations 5
usage: transformer_flops.py [-h]
                            (--hidden_size HIDDEN_SIZE [HIDDEN_SIZE ...] | --hidden_size_range HIDDEN_SIZE_RANGE [HIDDEN_SIZE_RANGE ...])
                            (--num_attention_heads NUM_ATTENTION_HEADS [NUM_ATTENTION_HEADS ...] | --num_attention_heads_range NUM_ATTENTION_HEADS_RANGE [NUM_ATTENTION_HEADS_RANGE ...])
                            (--vocab_size VOCAB_SIZE [VOCAB_SIZE ...] | --vocab_size_range VOCAB_SIZE_RANGE [VOCAB_SIZE_RANGE ...])
                            (--seq_length SEQ_LENGTH [SEQ_LENGTH ...] | --seq_length_range SEQ_LENGTH_RANGE [SEQ_LENGTH_RANGE ...])
                            (--microbatch_size MICROBATCH_SIZE [MICROBATCH_SIZE ...] | --microbatch_size_range MICROBATCH_SIZE_RANGE [MICROBATCH_SIZE_RANGE ...])
                            (--global_batch_size GLOBAL_BATCH_SIZE [GLOBAL_BATCH_SIZE ...] | --global_batch_size_range GLOBAL_BATCH_SIZE_RANGE [GLOBAL_BATCH_SIZE_RANGE ...])
                            (--tensor_mp_size TENSOR_MP_SIZE [TENSOR_MP_SIZE ...] | --tensor_mp_size_range TENSOR_MP_SIZE_RANGE [TENSOR_MP_SIZE_RANGE ...])
                            [--blocks BLOCKS [BLOCKS ...]] [--use_flash] [--num_iterations NUM_ITERATIONS]
                            [--num_warmup_iterations NUM_WARMUP_ITERATIONS] [--cuda_device CUDA_DEVICE] [--output_file OUTPUT_FILE]

options:
  -h, --help            show this help message and exit
  --hidden_size HIDDEN_SIZE [HIDDEN_SIZE ...]
                        The hidden dimension, enter any number of arguments
  --hidden_size_range HIDDEN_SIZE_RANGE [HIDDEN_SIZE_RANGE ...]
                        The hidden dimension, [start,stop,step]
  --num_attention_heads NUM_ATTENTION_HEADS [NUM_ATTENTION_HEADS ...]
                        The number of attention heads, enter any number of arguments
  --num_attention_heads_range NUM_ATTENTION_HEADS_RANGE [NUM_ATTENTION_HEADS_RANGE ...]
                        The number of attention heads, [start,stop,step]
  --vocab_size VOCAB_SIZE [VOCAB_SIZE ...]
                        The vocabulary size, enter any number of arguments
  --vocab_size_range VOCAB_SIZE_RANGE [VOCAB_SIZE_RANGE ...]
                        The vocabulary size, [start,stop,step]
  --seq_length SEQ_LENGTH [SEQ_LENGTH ...]
                        The sequence length, enter any number of arguments
  --seq_length_range SEQ_LENGTH_RANGE [SEQ_LENGTH_RANGE ...]
                        The sequence length, [start,stop,step]
  --microbatch_size MICROBATCH_SIZE [MICROBATCH_SIZE ...]
                        The microbatch size, enter any number of arguments
  --microbatch_size_range MICROBATCH_SIZE_RANGE [MICROBATCH_SIZE_RANGE ...]
                        The microbatch size, [start,stop,step]
  --global_batch_size GLOBAL_BATCH_SIZE [GLOBAL_BATCH_SIZE ...]
                        The global batch size, enter any number of arguments
  --global_batch_size_range GLOBAL_BATCH_SIZE_RANGE [GLOBAL_BATCH_SIZE_RANGE ...]
                        The global batch size, [start,stop,step]
  --tensor_mp_size TENSOR_MP_SIZE [TENSOR_MP_SIZE ...]
                        The tensor parallel size, enter any number of arguments
  --tensor_mp_size_range TENSOR_MP_SIZE_RANGE [TENSOR_MP_SIZE_RANGE ...]
                        The tensor parallel size, [start,stop,step]
  --blocks BLOCKS [BLOCKS ...]
                        The transformer blocks to benchmark, enter "all" or any number of [qkv_transform, attention_score,
                        attention_over_value, attention_linear_projection, mlp_h_to_4h, mlp_4h_to_h, logit_block, layer_norm, dropout,
                        add_bias_dropout, softmax, gelu]
  --use_flash           Use flash attention
  --num_iterations NUM_ITERATIONS
                        The number of iterations used to benchmark each BMM
  --num_warmup_iterations NUM_WARMUP_ITERATIONS
                        The number of warmup iterations
  --cuda_device CUDA_DEVICE
                        The cuda device to run the benchmark on
  --output_file OUTPUT_FILE
  --verbose, --no-verbose
                        log to stdout besides output_file? (default: True)
```

## Output Files
The output files will be in a text based format, and can be read into a `Pandas.dataframe`. An example of this is found in `plotting/transformer_figures.ipynb`. Alternatively, users can convert this output file into a csv using the `plotting/convert_to_csv` script.
Example:
```
python convert_to_csv.py --file_name ../results/bmm.out --output_file ../results/bmm.csv
```

