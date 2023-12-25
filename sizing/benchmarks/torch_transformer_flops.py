import time
import torch
import os
import numpy as np
import megatron_wrapper
import megatron
from megatron.model import LayerNorm
from megatron.model.fused_softmax import FusedScaleMaskSoftmax, SoftmaxFusionTypes
from megatron.model.transformer import ParallelSelfAttention, ParallelMLP, ParallelTransformerLayer
from megatron.model.transformer import bias_dropout_add_fused_train
from megatron.model.activations import bias_gelu_impl
from megatron.model.gpt2_model import gpt2_attention_mask_func as attention_mask_func
from megatron.model.word_embeddings import Embedding

print(torch.__version__, "\n")

dtype = torch.float16

def display(shape):
    return "x".join([str(dim) for dim in shape])


def initialize_mm_b(dtype, M, N, K, b):
    torch.cuda.empty_cache()
    sizes = [(b, M, K), (K, N), (b, M, N), (b, M, N)]
    return [torch.randint(-3, 3, size, device='cuda').to(dtype) for size in sizes]

def initialize_mm(dtype, M, N, K, b):
    torch.cuda.empty_cache()
    sizes = [(M, K), (K, N), (M, N), (M, N)]
    return [torch.randint(-3, 3, size, device='cuda').to(dtype) for size in sizes]

def initialize_bmm(dtype, M, N, K, b):
    torch.cuda.empty_cache()
    sizes = [(b, M, K), (b, K, N), (b, M, N), (b, M, N)]
    return [torch.randint(-3, 3, size, device='cuda').to(dtype) for size in sizes]


def benchmark_mm(m, n, k, label, b=None, num_iterations=200):
    B = torch.randn((k, n)).half().to("cuda:0")
    if b is None:
        A = torch.randn((m, n)).half().to("cuda:0")
        C = torch.empty((m, k)).half().to("cuda:0")
        b = 1
    else:
        A = torch.randn((b, m, n)).half().to("cuda:0")
        C = torch.empty((b, m, k)).half().to("cuda:0")
    num_warmup_iterations = 50
    times = np.zeros(num_iterations+num_warmup_iterations)
    start_time = time.time()
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            # torch.mm(A, B, out=C)
            torch.nn.functional.linear(A, B, out=C)
        torch.cuda.synchronize()
        times[i] = time.time()
    times -= start_time
    times = np.diff(times)
    times = times[50:]
    median_time = np.amax(times)
    print(f"Elapsed time for {label} ({m}x{n}x{k}, b={b}): {median_time:.4f}")
    print(f"Throughput (in TFLOP/s) for {label} ({m}x{n}x{k}, b={b}): "
          f"{(2 * b * m * n * k) / (median_time * 10**12):.3f}")
    return median_time

def benchmark_mm_cutlass(m,n,k,label,b=None, num_iterations=100):
    plan = cutlass.op.Gemm(element=dtype, layout=cutlass.LayoutType.RowMajor)
    if b is None:
        As, Bs, Cs, Ds, = initialize_mm(dtype, m, n, k, b)
        b=0
    else:
        As, Bs, Cs, Ds, = initialize_mm_b(dtype, m, n, k, b)
    torch.cuda.empty_cache()
    #print(torch.cuda.memory_summary(device='cuda'))
    num_warm = 50
    times = np.zeros(num_iterations+num_warm)
    start_time = time.time()
    for i in range( num_warm + num_iterations):
        plan.run(As, Bs, Cs, Ds, sync=True)
        torch.cuda.synchronize()
        times[i] = time.time()
    times -= start_time
    times = np.diff(times)
    times = times[50:]
    median_time = np.amax(times)
    del As, Bs, Cs, Ds
    print(f"Elapsed time for {label} ({m}x{n}x{k}, b={b}): {median_time:.4f}")
    print(f"Throughput (in TFLOP/s) for {label} ({m}x{n}x{k}, b={b}): "
          f"{(2 * b * m * n * k) / (median_time * 10**12):.3f}")
    return median_time

def benchmark_bmm(b, m, n, k, label, num_iterations=200):
    A = torch.randn((b, m, n)).half().to("cuda:0")
    B = torch.randn((b, n, k)).half().to("cuda:0")
    C = torch.empty((b, m, k)).half().to("cuda:0")
    num_warmup_iterations = 50

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
    median_time = np.amax(times)
    print(f"Elapsed time for {label} ({b}x{m}x{n}x{k}): {median_time:.4f}")
    print(f"Throughput (in TFLOP/s) for {label} ({b}x{m}x{n}x{k}): "
          f"{(2 * b * m * n * k) / (median_time * 10**12):.3f}")
    return median_time

def benchmark_bmm_cutlass(b, m, n, k, label, num_iterations=100):
    print(f"b: {b}, m: {m}, n: {n}, k: {k},")
    As, Bs, Cs, Ds, = initialize_bmm(dtype, m, n, k, b)
    plan = cutlass.op.Gemm(element=dtype, layout=cutlass.LayoutType.RowMajor)
    plan.compile()
    torch.cuda.empty_cache()
    num_warm = 50
    times = np.zeros(num_iterations+num_warm)
    start_time = time.time()
    for i in range( num_warm + num_iterations):
        plan.run(As, Bs, Cs, Ds, sync=True)
        torch.cuda.synchronize()
        times[i] = time.time()
    times -= start_time
    times = np.diff(times)
    times = times[50:]
    median_time = np.amax(times)
    del As, Bs, Cs, Ds
    torch.cuda.empty_cache()
    print(f"Elapsed time for {label} ({b}x{m}x{n}x{k}): {median_time:.4f}")
    print(f"Throughput (in TFLOP/s) for {label} ({b}x{m}x{n}x{k}): "
          f"{(2 * b * m * n * k) / (median_time * 10**12):.3f}")
    return median_time


def benchmark_dropout(A_dim, label, num_iterations=100):
    A = torch.randn(A_dim).half().to("cuda:0")
    dropout = torch.nn.Dropout(0.5).to("cuda:0")
    num_warmup_iterations = 50

    times = np.zeros(num_iterations+num_warmup_iterations)
    start_time = time.time()
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            dropout(A)
        torch.cuda.synchronize()
        times[i] = time.time()

    #elapsed_time = (time.time() - start_time) / num_iterations
    times -= start_time
    times = np.diff(times)
    times = times[50:]
    median_time = np.amax(times)

    '''for i in range(num_warmup_iterations + num_iterations):
        if i == num_warmup_iterations:
            start_time = time.time()
        with torch.no_grad():
            dropout(A)
        torch.cuda.synchronize()
    elapsed_time = (time.time() - start_time) / num_iterations'''
    print(f"Elapsed time for {label} ({display(A_dim)}): {median_time:.4f}")
    return median_time


def benchmark_softmax(scores_shape, seq_length, label, num_iterations=100):
    scores = torch.randn(scores_shape).half().to("cuda:0")
    attention_mask = torch.tril(torch.ones(
        (1, seq_length, seq_length), device="cuda:0")).view(
        1, 1, seq_length, seq_length)
    attention_mask = attention_mask < 0.5
    softmax = FusedScaleMaskSoftmax(
        True, False,
        SoftmaxFusionTypes.none, #attentionmasktype.padding=1,True
        attention_mask_func, True, 1)
    num_warmup_iterations = 50

    times = np.zeros(num_iterations+num_warmup_iterations)
    start_time = time.time()
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
           softmax(scores, attention_mask)
        torch.cuda.synchronize()
        times[i] = time.time()

    #elapsed_time = (time.time() - start_time) / num_iterations
    times -= start_time
    times = np.diff(times)
    times = times[50:]
    median_time = np.amax(times)

    '''for i in range(num_warmup_iterations + num_iterations):
        if i == num_warmup_iterations:
            start_time = time.time()
        with torch.no_grad():
            softmax(scores, attention_mask)
        torch.cuda.synchronize()
    elapsed_time = (time.time() - start_time) / num_iterations'''
    print(f"Elapsed time for {label} ({display(scores_shape)}): {median_time:.4f}")
    return median_time


def benchmark_fused_gelu(A_dim, b_dim, label, num_iterations=100):
    A = torch.randn(A_dim).half().to("cuda:0")
    b = torch.randn(b_dim).half().to("cuda:0")
    num_warmup_iterations = 50

    times = np.zeros(num_iterations+num_warmup_iterations)
    start_time = time.time()
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            bias_gelu_impl(A, b)
        torch.cuda.synchronize()
        times[i] = time.time()

    #elapsed_time = (time.time() - start_time) / num_iterations
    times -= start_time
    times = np.diff(times)
    times = times[50:]
    median_time = np.amax(times)

    '''for i in range(num_warmup_iterations + num_iterations):
        if i == num_warmup_iterations:
            start_time = time.time()
        with torch.no_grad():
            bias_gelu_impl(A, b)
        torch.cuda.synchronize()
    elapsed_time = (time.time() - start_time) / num_iterations'''
    print(f"Elapsed time for {label} ({display(A_dim)}): {median_time:.4f}")
    return median_time


def benchmark_layer_norm(A_dim, normalized_shape, label, num_iterations=100):
    A = torch.randn(A_dim).half().to("cuda:0")
    layer_norm = LayerNorm(normalized_shape).half().to("cuda:0")
    num_warmup_iterations = 50
    for i in range(num_warmup_iterations + num_iterations):
        if i == num_warmup_iterations:
            start_time = time.time()
        with torch.no_grad():
            layer_norm(A)
        torch.cuda.synchronize()
    elapsed_time = (time.time() - start_time) / num_iterations
    print(f"Elapsed time for {label} ({display(A_dim)}): {elapsed_time:.4f}")
    return elapsed_time


def benchmark_add_bias_dropout(shape, label, num_iterations=100):
    A = torch.randn(shape).half().to("cuda:0")
    bias = torch.randn(shape).half().to("cuda:0")
    residue = torch.randn(shape).half().to("cuda:0")
    num_warmup_iterations = 50

    times = np.zeros(num_iterations+num_warmup_iterations)
    start_time = time.time()
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            bias_dropout_add_fused_train(A, bias, residue, 0.0)
        torch.cuda.synchronize()
        times[i] = time.time()

    #elapsed_time = (time.time() - start_time) / num_iterations
    times -= start_time
    times = np.diff(times)
    times = times[50:]
    median_time = np.amax(times)  

    '''for i in range(num_warmup_iterations + num_iterations):
        if i == num_warmup_iterations:
            start_time = time.time()
        with torch.no_grad():
            bias_dropout_add_fused_train(A, bias, residue, 0.0)
        torch.cuda.synchronize()
    elapsed_time = (time.time() - start_time) / num_iterations'''
    print(f"Elapsed time for {label} ({display(shape)}): {median_time:.4f}")
    return median_time


def benchmark_transformer_from_mm_and_bmm(configuration, seq_length, global_batch_size, num_iterations=100):

    (microbatch_size, hidden_size, (tensor_mp_size, pipeline_mp_size, dp_size), num_attention_heads,vocab_size) = configuration
    print("\n\nEstimate")
    print("--------")
    elapsed_attention_time = 0.0
    elapsed_mlp_time = 0.0
    elapsed_add_bias_dropout_time = 0.0
    elapsed_layer_norm_time = 0.0
    '''elapsed_attention_time += benchmark_mm(
        microbatch_size, hidden_size,
        3 * hidden_size // tensor_mp_size,
        'attention_key_value_query_transform',
        b=seq_length, num_iterations=num_iterations)'''
    elapsed_attention_time += benchmark_bmm(
        microbatch_size * num_attention_heads // tensor_mp_size,
        seq_length, hidden_size // num_attention_heads,
        seq_length, 'attention_key_query_prob',
        num_iterations=num_iterations)
    elapsed_attention_time += benchmark_bmm(
        microbatch_size * num_attention_heads // tensor_mp_size,
        seq_length, seq_length, hidden_size // num_attention_heads,
        'attention_prob_times_values',
        num_iterations=num_iterations)
    '''
    elapsed_attention_time += benchmark_dropout(
        (microbatch_size, num_attention_heads // tensor_mp_size, seq_length, seq_length),
        'attention_dropout',
        num_iterations=num_iterations)
    elapsed_attention_time += benchmark_softmax(
        (microbatch_size, num_attention_heads // tensor_mp_size, seq_length, seq_length),
        seq_length, 'attention_softmax',
        num_iterations=num_iterations)
    '''
    '''
    elapsed_attention_time += benchmark_mm(
        microbatch_size, hidden_size // tensor_mp_size,
        hidden_size, 'attention_linear_projection',
        b=seq_length,
        num_iterations=num_iterations)
    
    elapsed_mlp_time += benchmark_mm(
        microbatch_size, hidden_size,
        4 * hidden_size // tensor_mp_size, 'mlp_h_to_4h',
        b=seq_length,
        num_iterations=num_iterations)
    '''
    '''
    elapsed_mlp_time += benchmark_fused_gelu(
        (seq_length, microbatch_size, 4 * hidden_size // tensor_mp_size),
        (4 * hidden_size // tensor_mp_size,),
        'mlp_fused_gelu', num_iterations=num_iterations)
    
    
    '''
    '''
    elapsed_mlp_time += benchmark_mm(
        microbatch_size, 4 * hidden_size // tensor_mp_size,
        hidden_size, 'mlp_4h_to_h',
        b=seq_length,
        num_iterations=num_iterations)
    '''
    '''
    elapsed_add_bias_dropout_time = 2 * benchmark_add_bias_dropout(
        (seq_length, microbatch_size, hidden_size),
        'transformer_add_bias_dropout',
        num_iterations=num_iterations)
    elapsed_layer_norm_time = 2 * benchmark_layer_norm(
        (seq_length, microbatch_size, hidden_size),
        hidden_size,
        'transformer_layer_norm',
        num_iterations=num_iterations)'''
    elapsed_total_time = elapsed_attention_time + elapsed_mlp_time + elapsed_add_bias_dropout_time + \
            elapsed_layer_norm_time

    num_attention_floating_point_operations =  \
        (4 * microbatch_size * seq_length * hidden_size / tensor_mp_size) * (
            2 * hidden_size + seq_length)
    num_mlp_floating_point_operations = \
        16 * microbatch_size * seq_length * hidden_size * hidden_size / tensor_mp_size
    num_total_floating_point_operations = num_attention_floating_point_operations + \
        num_mlp_floating_point_operations
    attention_throughput = num_attention_floating_point_operations / (elapsed_attention_time * 10**12)
    mlp_throughput = 1# num_mlp_floating_point_operations / (elapsed_mlp_time * 10**12)
    total_throughput = 1# num_total_floating_point_operations / (elapsed_total_time * 10**12)

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
    throughput = num_total_floating_point_operations / (elapsed_time * 10**12)


def benchmark_transformer(configuration, seq_length, global_batch_size, num_iterations=100):
    (microbatch_size, hidden_size,
     (tensor_mp_size, pipeline_mp_size, dp_size), num_attention_heads,vocab_size) = configuration
    print("\n\nActual")
    print("------")

    args = megatron_wrapper.get_megatron_args(configuration)
    fn_args = [megatron.model.init_functions.init_method_normal(args.init_method_std),
               megatron.model.init_functions.init_method_normal(args.init_method_std)]
    init_method = megatron.model.init_functions.init_method_normal(args.init_method_std)
    #embedding_layer = Embedding(args,hidden_size,vocab_size,seq_length,0.0,init_method=init_method,use_pos_emb=False)
    args.attention_config=["flash","global"]
    attention_layer = ParallelSelfAttention(args,attention_mask_func=attention_mask_func, init_method=init_method,output_layer_init_method=init_method, layer_number=0).half().to("cuda:0")
    mlp_layer = ParallelMLP(args,init_method=init_method,output_layer_init_method=init_method).half().to("cuda:0")
    transformer_layer = ParallelTransformerLayer(args,attention_mask_func=attention_mask_func,init_method=init_method,output_layer_init_method=init_method,layer_number=0).half().to("cuda:0")
    inp = torch.randn((args.seq_length, args.batch_size, args.hidden_size)).half().to("cuda:0")
    attention_mask = torch.tril(torch.ones(
        (1, args.seq_length, args.seq_length), device="cuda:0")).view(
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

    num_warmup_iterations = 50
    allTimes = []
    for layer, label, need_attention_mask, num_floating_point_operations in \
        zip([ attention_layer, mlp_layer, transformer_layer],
            [ "Attention", "MLP", "Transformer"],
            [ True, False, True],
            [num_attention_floating_point_operations, num_mlp_floating_point_operations,
             num_total_floating_point_operations]):
        layer.train()

        times = np.zeros(num_iterations+num_warmup_iterations)
        start_time = time.time()
        for i in range(num_warmup_iterations + num_iterations):
            with torch.no_grad():
                if need_attention_mask:
                    out = layer(inp, attention_mask)
                    torch.cuda.empty_cache()
                else:
                    out = layer(inp)

            torch.cuda.synchronize()
            times[i] = time.time()

        #elapsed_time = (time.time() - start_time) / num_iterations
        times -= start_time
        times = np.diff(times)
        times = times[50:]
        median_time = np.median(times) 

        '''for i in range(num_warmup_iterations + num_iterations):
            if i == num_warmup_iterations:
                start_time = time.time()
            with torch.no_grad():
                if need_attention_mask:
                    out = layer(inp, attention_mask)
                else:
                    out = layer(inp)
            torch.cuda.synchronize()
        elapsed_time = (time.time() - start_time) / num_iterations'''
        allTimes.append(median_time)
        throughput = num_floating_point_operations / (median_time * 10**12)
        print(f"{label} duration (in seconds): {median_time:.4f}")
        print(f"{label} throughput (in TFLOP/s): {throughput:.3f}")
    #print("Transformer - MLP - Attention (in seconds): "f"{(allTimes[-1] - allTimes[0] - allTimes[1]):.4f}")


if __name__ == '__main__':
    torch.cuda.set_device("cuda:0")

    seq_length = 2048
    train_batch_size = 2048
    configurations = []
    for tensor_mp_size in [1]:
        for num_attention_heads in [128]: # [32,128]: #[32, 64, 96, 128]:
            for hidden_size in [16384]:#range(128,2**15,128): # range(num_attention_heads*8,2**15 + num_attention_heads,num_attention_heads*8): #[32768]: #range(8192,2**15, num_attention_heads):
                for microbatch_size in [4]:
                    for vocab_size in [51200]:
                        configurations.append((microbatch_size, hidden_size,
                                           (tensor_mp_size, 1, 1), num_attention_heads,vocab_size))
    megatron_wrapper.initialize_megatron(configurations[0])
    for configuration in configurations:
        (microbatch_size, hidden_size,
                (tensor_mp_size, pipeline_mp_size, dp_size), num_attention_heads,vocab_size) = configuration
        label = {'num_attention_heads': num_attention_heads,
                 'hidden_size': hidden_size,
                 'train_micro_batch_size_per_gpu': microbatch_size,
                 'tensor_mp_size': tensor_mp_size,
                 'pipeline_mp_size': pipeline_mp_size,
                 'dp_size': dp_size}
        label_str = ", ".join([f"{k}: {v}" for (k, v) in label.items()])
        print(label_str)
        #benchmark_transformer_from_mm_and_bmm(configuration, seq_length, train_batch_size)
        benchmark_transformer(configuration, seq_length, train_batch_size)
        print("=" * 120)
