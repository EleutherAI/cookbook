# By Quentin Anthony and Hailey Schoelkopf

import argparse
import math

# Helper function to pretty-print message sizes
def convert_params(params):
    if params == 0:
        return "0"
    size_name = ("", "K", "M", "B", "T", "P", "E", "Z", "Y")
    i = int(math.floor(math.log(params, 1000)))
    p = math.pow(1000, i)
    s = round(params / p, 2)
    return "%s %s" % (s, size_name[i])

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", "-p",
                        type=int,
                        default=20000000000,
                        help='Number of Parameters')
    parser.add_argument("--num-gpus",
                        type=int,
                        default=1,
                        help='Number of GPUs used for training')
    parser.add_argument("--tensor-parallel-size", "-tp",
                        type=int,
                        default=1,
                        help='Tensor parallel degree (1 if not used)')
    parser.add_argument("--pipeline-parallel-size", "-pp",
                        type=int,
                        default=1,
                        help='Pipeline parallel degree (1 if not used)')
    parser.add_argument("--partition-activations", "-pa",
                        action="store_true",
                        help='Whether we use ZeRO-R to partition activation memory across tensor-parallel degree')
    parser.add_argument("--zero-stage", "-z",
                        type=int,
                        default=1,
                        choices=[0,1,2,3],
                        help='Stage of the ZeRO optimizer')
    parser.add_argument("--checkpoint-activations", "-ca",
                        action="store_true",
                        help='Whether Megatron-style activation checkpointing is being used')
    parser.add_argument("--batch-size-per-gpu", "-b",
                        type=int,
                        default=1,
                        help='Batch size per GPU')
    parser.add_argument("--hidden-size", "-hs",
                        type=int,
                        default=6144,
                        help='Dimension of the model\'s hidden size')
    parser.add_argument("--num-attention-heads", "-a",
                        type=int,
                        default=64,
                        help='Number of attention heads used in model')
    parser.add_argument("--kv-size-ratio", "-kv",
                        type=float,
                        default=1.0,
                        help='Ratio of total query heads to key/value heads. 1.0 for MHA, 1/num_attention_heads for MQA.')
    parser.add_argument("--sequence-length", "-s",
                        type=int,
                        default=2048,
                        help='Sequence length used for training')
    parser.add_argument("--num-layers", "-l",
                        type=int,
                        default=44,
                        help='Number of transformer layers used in model')
    parser.add_argument("--fp32-model",
                        action="store_true",
                        help='Whether model is stored in fp32')
    parser.add_argument("--fp32-grads",
                        action="store_true",
                        help='Whether grads are stored in fp32')
    parser.add_argument("--zero-allgather-bucket-size", "-zbs",
                        type=int,
                        default=5e8,
                        help='Size of allgather buckets used by ZeRO')
    parser.add_argument("--zero3-max-live-params", "-zmlp",
                        type=int,
                        default=1e9,
                        help='Maximum number of parameters ZeRO3 keeps in GPU memory')
    parser.add_argument("--misc-mem-gb",
                        type=int,
                        default=0,
                        help='Miscellaneous memory overhead by DL framework(s), communication libraries, etc')
    parser.add_argument("--num-experts",
                    type=int,
                    default=0,
                    help='Number of experts')
    parser.add_argument("--ffn-expansion-factor", "-ff",
                    type=int,
                    default=4,
                    help='How much the MLP hidden size expands')
    parser.add_argument("--expert-parallelism", "-ep",
                type=int,
                default=1,
                help='How many ways are the experts sharded across ranks')
    parser.add_argument("--vocab-size", "-v",
            type=int,
            default=51200,
            help='How many ways are the experts sharded across ranks')
    return parser

# calculates the total memory necessary for training a model
def calc_mem(args):

    dp_degree = args.num_gpus / (args.tensor_parallel_size * args.pipeline_parallel_size)

    # 4 bytes in fp32, 2 bytes in fp16/bf16
    if args.fp32_model:
        bytes_per_param = 4
    else:
        bytes_per_param = 2


    # compute total parameters from the config
    embed_params = 2 * args.vocab_size * args.hidden_size
    positional_params = args.hidden_size * args.sequence_length
    ln_params = 8 * args.hidden_size * args.num_layers + (2 * args.hidden_size)
    attention_params = int(2 * (1 + args.kv_size_ratio) * args.num_layers * args.hidden_size * args.hidden_size)
    mlp_params = 2 * args.num_layers * args.hidden_size * args.ffn_expansion_factor * args.hidden_size
    total_params = embed_params + positional_params + ln_params + attention_params + mlp_params
    if args.num_experts > 0:
        total_moe_params = embed_params + positional_params + ln_params + attention_params + (args.num_experts * mlp_params)
    # Split the model with 3D parallelism
    if args.num_experts == 0:
        model_mem = (total_params * bytes_per_param) / (args.tensor_parallel_size * args.pipeline_parallel_size)
    else:
        EP_total_params = embed_params + positional_params + ln_params + attention_params + ((args.num_experts/args.expert_parallelism) * mlp_params)
        model_mem = (EP_total_params * bytes_per_param) / (args.tensor_parallel_size * args.pipeline_parallel_size)
    # ZeRO stage 3 shards the model parameters across GPUs (plus the gradients and optimizer states)
    if args.zero_stage == 3:
        model_mem /= args.num_gpus

    # 4 bytes in fp32, 2 bytes in fp16/bf16
    if args.fp32_grads:
        bytes_per_grad_element = 4
    else:
        bytes_per_grad_element = 2

    if args.num_experts > 0:
        gradient_mem = EP_total_params * bytes_per_grad_element
    else:
        gradient_mem = total_params * bytes_per_grad_element
    # ZeRO stage 2 shards the gradients across GPUs (plus the optimizer states)
    if args.zero_stage >= 2:
        gradient_mem /= args.num_gpus
    gradient_mem /= args.pipeline_parallel_size

    # For mixed-precision Adam/AdamW, the optimizer must store fp32 copies of the parameters, momentum, and variance (4 + 4 + 4 = 12 bytes per optimizer parameter)
    # Feel free to change the multiplier for your optimizer (examples include SGD (4 + 4 = 8) and 8-bit ADAM (2 + 2 + 2 = 6)
    if args.num_experts > 0:
        optimizer_mem = EP_total_params * 12
    else:
        optimizer_mem = total_params * 12
    # ZeRO stage 3 shards the optimizer states across GPUs
    if args.zero_stage >= 1:
        optimizer_mem /= args.num_gpus

    communication_mem = 0
    # The size of the communication buffer DeepSpeed uses to store ZeRO optimizer elements
    if args.zero_stage >= 1:
        communication_mem += args.zero_allgather_bucket_size * bytes_per_param
    # The number of parameters ZeRO-3 keeps alive in GPU memory at a time
    if args.zero_stage == 3:
        communication_mem += args.zero3_max_live_params * bytes_per_param

    # Taken from Table 2 in https://arxiv.org/pdf/1910.02054.pdf
    # We find these don't perfectly match with experiment, but are good approximations
    if args.checkpoint_activations:
        activation_mem = args.sequence_length * args.batch_size_per_gpu * args.hidden_size * args.num_layers * (10 + (24 / args.tensor_parallel_size))
    else:
        activation_mem = args.sequence_length * args.batch_size_per_gpu * args.hidden_size * args.num_layers * (10 + (24 / args.tensor_parallel_size) + 5 * ((args.num_attention_heads * args.sequence_length) / (args.hidden_size * args.tensor_parallel_size)))

    # DeepSpeed's ZeRO-R partitions activation memory across tensor-parallel GPUs
    if args.partition_activations:
        activation_mem /= args.tensor_parallel_size


    # We include a "Miscellaneous Memory" term because we find some 3D-parallel frameworks add a constant memory overhead (~5GB in our experiments with Megatron-DeepSpeed) that we cannot explain. If you know the source of this, add a comment!
    gradient_mem_gb = gradient_mem / 1024**3
    activation_mem_gb = activation_mem / 1024**3
    model_mem_gb = model_mem / 1024**3
    optimizer_mem_gb = optimizer_mem / 1024**3
    communication_mem_gb = communication_mem / 1024**3
    total_mem_gb = activation_mem_gb + gradient_mem_gb + model_mem_gb + optimizer_mem_gb + communication_mem_gb + args.misc_mem_gb
    print(f'Calculating memory with training configuration: {vars(args)}\n')
    print(f'Number of Parameters: {convert_params(total_params)}')
    if args.num_experts > 0:
        print(f'Total Number of MoE Parameters: {convert_params(total_moe_params)}')
    print(f'Gradient Memory: {gradient_mem_gb:.2f} GB')
    print(f'Activation Memory: {activation_mem_gb:.2f} GB')
    print(f'Model Memory: {model_mem_gb:.2f} GB')
    print(f'Optimizer Memory: {optimizer_mem_gb:.2f} GB')
    print(f'Communication Memory: {communication_mem_gb:.2f} GB')
    print(f'Miscellaneous Memory: {args.misc_mem_gb:.2f} GB')
    print(f'Total Memory Required for Training: {total_mem_gb:.2f} GB')

if __name__ == "__main__":
    print('\nExample with pythia 6.9B: python transformer_mem.py --num-layers=32 --sequence-length=2048 --num-attention-heads=32 --hidden-size=4096 --batch-size-per-gpu=8 --checkpoint-activations --zero-stage=1 --partition-activations --pipeline-parallel-size=1 --tensor-parallel-size=2 --num-gpus=128 --params=6900000000')
    print('Example with pythia 12B: python transformer_mem.py --num-layers=36 --sequence-length=2048 --num-attention-heads=40 --hidden-size=5120 --batch-size-per-gpu=8 --checkpoint-activations --zero-stage=1 --partition-activations --pipeline-parallel-size=1 --tensor-parallel-size=4 --num-gpus=256 --params=11849420800')
    print('Example with default 20B: python transformer_mem.py --num-layers=44 --sequence-length=2048 --num-attention-heads=64 --hidden-size=6144 --batch-size-per-gpu=1 --checkpoint-activations --zero-stage=1 --partition-activations --pipeline-parallel-size=1 --tensor-parallel-size=1 --num-gpus=1 --params=20000000000\n')
    args = config_parser().parse_args()
    calc_mem(args)
