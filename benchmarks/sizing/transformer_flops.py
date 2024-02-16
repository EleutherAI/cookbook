import time
import torch
import os
import numpy as np
import megatron_wrapper
import megatron
from megatron.model.transformer import ParallelSelfAttention, ParallelMLP, ParallelTransformerLayer
from megatron.model.gpt2_model import gpt2_attention_mask_func as attention_mask_func
import sys
from utils import *
import argparse
from megatron.model import LayerNorm
from megatron.model.fused_softmax import FusedScaleMaskSoftmax, SoftmaxFusionTypes
from megatron.model.transformer import ParallelSelfAttention, ParallelMLP, ParallelTransformerLayer
from megatron.model.transformer import bias_dropout_add_fused_train
from megatron.model.activations import bias_gelu_impl
from megatron.model.gpt2_model import gpt2_attention_mask_func as attention_mask_func
from megatron.model.word_embeddings import Embedding

file_dir = os.path.abspath(os.path.dirname(__file__))

# benchmarks the individual components of the transformer.  Will only be used if --layers is specified and will only benchmark the layers specified
def benchmark_transformer_from_mm_and_bmm(args, configuration, seq_length, global_batch_size, num_iterations, num_warmup_iterations):

    (microbatch_size, hidden_size, (tensor_mp_size, pipeline_mp_size, dp_size), num_attention_heads,vocab_size,seq_length,train_batch_size) = configuration
    print("\n\nEstimate")
    print("--------")
    elapsed_attention_time = 0.0
    elapsed_mlp_time = 0.0
    elapsed_add_bias_dropout_time = 0.0
    elapsed_layer_norm_time = 0.0
    attention_throughput = 0.0
    mlp_throughput = 0.0
    total_throughput = 0.0
    
    if 'qkv_transform' in args.blocks or 'all' in args.blocks:
        elapsed_attention_time += benchmark_mm_b(
            microbatch_size, hidden_size,
            3 * hidden_size // tensor_mp_size,
            'qkv_transform',
            seq_length, num_iterations, num_warmup_iterations)
    if 'attention_score' in args.blocks or 'all' in args.blocks:
        elapsed_attention_time += benchmark_bmm(
            microbatch_size * num_attention_heads // tensor_mp_size,
            seq_length, hidden_size // num_attention_heads,
            seq_length, 'attention_score',
            num_iterations, num_warmup_iterations)
    if 'attention_over_value' in args.blocks or 'all' in args.blocks:
        elapsed_attention_time += benchmark_bmm(
            microbatch_size * num_attention_heads // tensor_mp_size,
            seq_length, seq_length, hidden_size // num_attention_heads,
            'attention_over_value',
            num_iterations, num_warmup_iterations)
    if 'dropout' in args.blocks or 'all' in args.blocks:
        elapsed_attention_time += benchmark_dropout(
            (microbatch_size, num_attention_heads // tensor_mp_size, seq_length, seq_length),
            'attention_dropout',
            num_iterations, num_warmup_iterations)
    if 'softmax' in args.blocks or 'all' in args.blocks:
        elapsed_attention_time += benchmark_softmax(
            (microbatch_size, num_attention_heads // tensor_mp_size, seq_length, seq_length),
            seq_length, 'attention_softmax',
            num_iterations, num_warmup_iterations)
    if 'attention_linear_projection' in args.blocks or 'all' in args.blocks:
        elapsed_attention_time += benchmark_mm_b(
            microbatch_size, hidden_size // tensor_mp_size,
            hidden_size, 'attention_linear_projection',
            seq_length,
            num_iterations, num_warmup_iterations)
    if 'mlp_h_to_4h' in args.blocks or 'all' in args.blocks:
        elapsed_mlp_time += benchmark_mm_b(
            microbatch_size, hidden_size,
            4 * hidden_size // tensor_mp_size, 'mlp_h_to_4h',
            seq_length,
            num_iterations, num_warmup_iterations)
    if 'gelu' in args.blocks or 'all' in args.blocks:
        elapsed_mlp_time += benchmark_fused_gelu(
            (seq_length, microbatch_size, 4 * hidden_size // tensor_mp_size),
            (4 * hidden_size // tensor_mp_size,),
            'mlp_fused_gelu', num_iterations, num_warmup_iterations)
    if 'mlp_4h_to_h' in args.blocks or 'all' in args.blocks:
        elapsed_mlp_time += benchmark_mm_b(
            microbatch_size, 4 * hidden_size // tensor_mp_size,
            hidden_size, 'mlp_4h_to_h',
            seq_length,
            num_iterations, num_warmup_iterations)
    if 'add_bias_dropout' in args.blocks or 'all' in args.blocks:
        elapsed_add_bias_dropout_time = 2 * benchmark_add_bias_dropout(
            (seq_length, microbatch_size, hidden_size),
            'transformer_add_bias_dropout',
            num_iterations, num_warmup_iterations)
    if 'layer_norm' in args.blocks or 'all' in args.blocks:
        elapsed_layer_norm_time = 2 * benchmark_layer_norm(
            (seq_length, microbatch_size, hidden_size),
            hidden_size,
            'transformer_layer_norm',
            num_iterations, num_warmup_iterations)
    if 'logit_block' in args.blocks or 'all' in args.blocks:
        elapsed_attention_time += benchmark_mm_b(
            microbatch_size, vocab_size,
            hidden_size, 'logit_block',
            seq_length,
            num_iterations, num_warmup_iterations)
    
    elapsed_total_time = elapsed_attention_time + elapsed_mlp_time + elapsed_add_bias_dropout_time + \
            elapsed_layer_norm_time

    num_attention_floating_point_operations =  \
        (4 * microbatch_size * seq_length * hidden_size / tensor_mp_size) * (
            2 * hidden_size + seq_length)
    num_mlp_floating_point_operations = \
        16 * microbatch_size * seq_length * hidden_size * hidden_size / tensor_mp_size
    num_total_floating_point_operations = num_attention_floating_point_operations + \
        num_mlp_floating_point_operations
    if elapsed_attention_time > 0:
        attention_throughput = num_attention_floating_point_operations / (elapsed_attention_time * 10**12)
    if elapsed_mlp_time > 0:
        mlp_throughput = num_mlp_floating_point_operations / (elapsed_mlp_time * 10**12)
    if elapsed_total_time > 0:
        total_throughput = num_total_floating_point_operations / (elapsed_total_time * 10**12)

    print()
    for (elapsed_time, throughput, label) in \
        zip([elapsed_attention_time, elapsed_mlp_time, elapsed_total_time],
            [attention_throughput, mlp_throughput, total_throughput],
            ["Attention", "MLP", "Transformer"]):
        print(f"{label} duration (in seconds): {elapsed_time:.4f}")
        print(f"{label} throughput (in TFLOP/s): {throughput:.3f}")
    print("Transformer - MLP - Attention (in seconds): "
          f"{(elapsed_total_time - elapsed_attention_time - elapsed_mlp_time):.4f}")

    num_microbatches_in_pipeline = global_batch_size // (microbatch_size * dp_size)
    pipeline_bubble_fraction = (pipeline_mp_size - 1) / num_microbatches_in_pipeline
    elapsed_time *= (1 + pipeline_bubble_fraction)
    # Throughput if considering pipeline bubble.
    throughput = num_total_floating_point_operations / (elapsed_time * 10**12 / 10**3)

# benchmarks the entire transformer using megatron
def benchmark_transformer(c_args,configuration, seq_length, global_batch_size, num_iterations,num_warmup_iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    (microbatch_size, hidden_size,
     (tensor_mp_size, pipeline_mp_size, dp_size), num_attention_heads,vocab_size,seq_length,train_batch_size) = configuration
    print("\n\nActual")
    print("------")

    args = megatron_wrapper.get_megatron_args(configuration)
    fn_args = [megatron.model.init_functions.init_method_normal(args.init_method_std),
               megatron.model.init_functions.init_method_normal(args.init_method_std)]
    init_method = megatron.model.init_functions.init_method_normal(args.init_method_std)
    if c_args.use_flash:
        args.attention_config=["flash","global"]
    attention_layer = ParallelSelfAttention(args,attention_mask_func=attention_mask_func, init_method=init_method,output_layer_init_method=init_method, layer_number=0).half().to("cuda")
    mlp_layer = ParallelMLP(args,init_method=init_method,output_layer_init_method=init_method).half().to("cuda")
    transformer_layer = ParallelTransformerLayer(args,attention_mask_func=attention_mask_func,init_method=init_method,output_layer_init_method=init_method,layer_number=0).half().to("cuda")
    inp = torch.randn((args.seq_length, args.batch_size, args.hidden_size)).half().to("cuda")
    attention_mask = torch.tril(torch.ones(
        (1, args.seq_length, args.seq_length), device="cuda")).view(
        1, 1, args.seq_length, args.seq_length)
    attention_mask = attention_mask < 0.5

    num_embedding_floating_point_operations = \
        (2*vocab_size -1) * seq_length * microbatch_size * hidden_size
    num_attention_floating_point_operations =  \
        (4 * microbatch_size * seq_length * hidden_size / tensor_mp_size) * (
            2 * hidden_size + seq_length)
    num_mlp_floating_point_operations = \
        16 * microbatch_size * seq_length * hidden_size * hidden_size / tensor_mp_size
    num_total_floating_point_operations = num_attention_floating_point_operations + \
        num_mlp_floating_point_operations
    
    for layer, label, need_attention_mask, num_floating_point_operations in \
        zip([ attention_layer, mlp_layer, transformer_layer],
            [ "Attention", "MLP", "Transformer"],
            [ True, False, True],
            [num_attention_floating_point_operations, num_mlp_floating_point_operations,
             num_total_floating_point_operations]):
        layer.train()
        
        times = np.zeros(num_iterations+num_warmup_iterations)
        for i in range(num_warmup_iterations + num_iterations):
            with torch.no_grad():
                start.record()
                if need_attention_mask:
                    out = layer(inp, attention_mask)
                    torch.cuda.empty_cache()
                else:
                    out = layer(inp)
                end.record()
            torch.cuda.synchronize()
            times[i] = start.elapsed_time(end)

        times = times[num_warmup_iterations:]
        elapsed_time = np.amax(times)/1000 # get to seconds from milliseconds

        throughput = num_floating_point_operations / (elapsed_time * 10**12)
        print(f"{label} duration (in seconds): {elapsed_time:.4f}")
        print(f"{label} throughput (in TFLOP/s): {throughput:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    h_group = parser.add_mutually_exclusive_group(required=True)
    h_group.add_argument("--hidden_size", nargs="+", type=int, help='The hidden dimension, enter any number of arguments')
    h_group.add_argument("--hidden_size_range", nargs='+', type=int, help="The hidden dimension, [start,stop,step]")

    a_group = parser.add_mutually_exclusive_group(required=True)
    a_group.add_argument("--num_attention_heads", nargs="+", type=int, help='The number of attention heads, enter any number of arguments')
    a_group.add_argument("--num_attention_heads_range", nargs='+', type=int, help="The number of attention heads, [start,stop,step]")

    v_group = parser.add_mutually_exclusive_group(required=True)
    v_group.add_argument("--vocab_size", nargs="+", type=int, help='The vocabulary size, enter any number of arguments')
    v_group.add_argument("--vocab_size_range", nargs='+', type=int, help="The vocabulary size, [start,stop,step]")

    s_group = parser.add_mutually_exclusive_group(required=True)
    s_group.add_argument("--seq_length", nargs="+", type=int, help='The sequence length, enter any number of arguments')
    s_group.add_argument("--seq_length_range", nargs='+', type=int, help="The sequence length, [start,stop,step]")

    b_group = parser.add_mutually_exclusive_group(required=True)
    b_group.add_argument("--microbatch_size", nargs="+", type=int, help='The microbatch size, enter any number of arguments')
    b_group.add_argument("--microbatch_size_range", nargs='+', type=int, help="The microbatch size, [start,stop,step]")

    gb_group = parser.add_mutually_exclusive_group(required=True)
    gb_group.add_argument("--global_batch_size", nargs="+", type=int, help='The global batch size, enter any number of arguments')
    gb_group.add_argument("--global_batch_size_range", nargs='+', type=int, help="The global batch size, [start,stop,step]")

    t_group = parser.add_mutually_exclusive_group(required=True)
    t_group.add_argument("--tensor_mp_size", nargs="+", type=int, help='The tensor parallel size, enter any number of arguments')
    t_group.add_argument("--tensor_mp_size_range", nargs='+', type=int, help="The tensor parallel size, [start,stop,step]")

    parser.add_argument("--blocks", nargs="+", type=str, help='The transformer blocks to benchmark, enter "all" or any number of [qkv_transform, attention_score, \
                          attention_over_value, attention_linear_projection, mlp_h_to_4h, mlp_4h_to_h, logit_block, layer_norm, dropout, add_bias_dropout, softmax, gelu]')

    parser.add_argument("--use_flash", action="store_true", help="Use flash  attention")
    parser.add_argument("--num_iterations", type=int, default=200, help='The number of iterations used to benchmark each BMM')
    parser.add_argument("--num_warmup_iterations", type=int, default=50, help='The number of warmup iterations')
    parser.add_argument("--cuda_device", type=int, default=0, help="The cuda device to run the benchmark on")
    parser.add_argument("--output_file", type=str, default=f"{file_dir}/results/mm.out")
    parser.add_argument("--verbose", default=True, action=argparse.BooleanOptionalAction, help='log to stdout besides output_file?')
    args = parser.parse_args()

    h = args.hidden_size
    a = args.num_attention_heads
    v = args.vocab_size
    s = args.seq_length
    t = args.tensor_mp_size
    b = args.microbatch_size
    global_batch_size = args.global_batch_size

    if h is None:
        start,stop,step = args.hidden_size_range
        h = np.arange(start,stop,step)
    if a is None:
        start,stop,step = args.num_attention_heads_range
        a = np.arange(start,stop,step)
    if v is None:
        start,stop,step = args.vocab_size_range
        v = np.arange(start,stop,step)
    if s is None:
        start,stop,step = args.seq_length_range
        s = np.arange(start,stop,step)
    if t is None:
        start,stop,step = args.tensor_mp_size_range
        t = np.arange(start,stop,step)
    if b is None:
        start,stop,step = args.microbatch_size_range
        b = np.arange(start,stop,step)
    if global_batch_size is None:
        start,stop,step = args.global_batch_size_range
        global_batch_size = np.arange(start,stop,step)

    torch.cuda.set_device(f"cuda:{args.cuda_device}")

    sys.stdout = Tee(args.output_file, args.verbose)

    configurations = []
    for train_batch_size in global_batch_size:
        for seq_length in s:
            for tensor_mp_size in t:
                for num_attention_heads in a:
                    for hidden_size in h:
                        for microbatch_size in b:
                            for vocab_size in v:
                                configurations.append((microbatch_size, hidden_size,
                                        (tensor_mp_size, 1, 1), num_attention_heads,vocab_size,seq_length,train_batch_size))
        #megatron_wrapper.initialize_megatron(configurations[0])
        for configuration in configurations:
            (microbatch_size, hidden_size,
                    (tensor_mp_size, pipeline_mp_size, dp_size), num_attention_heads,vocab_size,seq_length,train_batch_size) = configuration
            label = {'num_attention_heads': num_attention_heads,
                    'hidden_size': hidden_size,
                    'train_micro_batch_size_per_gpu': microbatch_size,
                    'seq_length': seq_length,
                    'vocab_size': vocab_size,
                    'train_batch_size': train_batch_size,
                    'tensor_mp_size': tensor_mp_size,
                    'pipeline_mp_size': pipeline_mp_size,
                    'dp_size': dp_size}
            label_str = ", ".join([f"{k}: {v}" for (k, v) in label.items()])
            print(label_str)
            if args.blocks is None:
                benchmark_transformer(args,configuration, seq_length, train_batch_size, args.num_iterations, args.num_warmup_iterations)
            else:
                benchmark_transformer_from_mm_and_bmm(args,configuration, seq_length, train_batch_size, args.num_iterations, args.num_warmup_iterations)
            print("=" * 120)
