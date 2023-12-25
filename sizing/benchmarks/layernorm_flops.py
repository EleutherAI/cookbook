import time
import torch
import megatron
from megatron.model import LayerNorm

def display(shape):
    return "x".join([str(dim) for dim in shape])

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
    throughput = 2 * A.numel() * 1e-9 / elapsed_time
    print(f"Elapsed time for {label} ({display(A_dim)}): {elapsed_time}")
    print(f"Throughput (in TFLOP/s) for {label} ({display(A_dim)}): {throughput}")
    return elapsed_time 

if __name__ == '__main__':
    torch.cuda.set_device("cuda:0")

    seq_length=2048
    microbatch_size=4
    for hidden_size in range(14336-128,14336+128+1):
        benchmark_layer_norm(
            (seq_length, microbatch_size, hidden_size),
            hidden_size,
            'transformer_layer_norm')
    hidden_size = 14336
    for microbatch_size in range(8):
        benchmark_layer_norm(
            (seq_length, microbatch_size, hidden_size),
            hidden_size,
            'transformer_layer_norm')
    microbatch_size=4
    for seq_length in range(2048-128,2048+128+1):
        benchmark_layer_norm(
            (seq_length, microbatch_size, hidden_size),
            hidden_size,
            'transformer_layer_norm')