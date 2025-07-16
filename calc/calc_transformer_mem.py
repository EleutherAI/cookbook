# By Quentin Anthony, Hailey Schoelkopf, Bhavnick Minhas

import argparse
import math


### Begin Helper Functions ###

def convert_params(params):
    '''
    Helper function to pretty-print message sizes
    '''
    if params == 0:
        return "0"
    size_name = ("", "K", "M", "B", "T", "P", "E", "Z", "Y")
    i = int(math.floor(math.log(params, 1000)))
    p = math.pow(1000, i)
    s = round(params / p, 2)
    return "%s %s" % (s, size_name[i])

def set_defaults(args):
    '''
    Sets the default values for the arguments that are not provided
    '''
    for key, value in DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args

def set_if_none(args, key, config, config_key):
    '''
    Sets the value of the argument to the default value if it is not provided
    '''
    if getattr(args, key) is None:
        setattr(args, key, config.get(config_key, DEFAULTS[key]))
    else:
        print(f"overriding HF {config_key} config value ({config[config_key]}) with provided value ({getattr(args, key)})")
    return args

def get_hf_model_args(args):
    '''
    Updates the args with HuggingFace model config values
    '''
    # Check if the name is not None
    if args.hf_model_name_or_path is not None:
        try: 
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(args.hf_model_name_or_path, 
                                                trust_remote_code=True).to_dict()
        except OSError as e:
            print("An OSError has been raised. Commonly due to a model Repository name or path not found. Are you sure it exists?")
            print('Full error: ')
            raise e
        except ImportError as e:
            print('If you would like to calculate from a HF model, you must install HF transformers with pip install transformers')
            print('Full error: ')
            raise e
        
        # Now that config has been retrieved, we update the args with the config values

        arch = config['model_type']

       # Seperate handling for gpt2 because they named everything differently 
        if arch.lower()=='gpt2': 
            args.num_layers = config.get("n_layer", args.num_layers) 
            args.num_attention_heads = config.get("n_head", args.num_attention_heads)
            args.hidden_size = config.get("n_embd", args.hidden_size)
            args.vocab_size = config.get("vocab_size", args.vocab_size)

        else:  
            set_if_none(args, "num_layers", config, "num_hidden_layers")
            set_if_none(args, "num_attention_heads", config, "num_attention_heads")
            set_if_none(args, "hidden_size", config, "hidden_size")

            config["ffn_expansion_factor"] = config.get("intermediate_size", args.hidden_size) / args.hidden_size
            set_if_none(args, "ffn_expansion_factor", config, "ffn_expansion_factor")

            # config["num_key_value_heads"] = config.get("num_key_value_heads", config["num_attention_heads"])
            # set_if_none(args, "num_key_value_heads", config, "num_key_value_heads")

            set_if_none(args, "vocab_size", config, "vocab_size")
            set_if_none(args, "sequence_length", config, "max_position_embeddings")
            
    # Set the default values regardless
    set_defaults(args)

    return args

### End Helper Functions ###

### Begin Argument Parsing ###

def config_parser():
    parser = argparse.ArgumentParser()
    # HuggingFace Settings
    parser.add_argument("--hf_model_name_or_path", 
                        type=str, 
                        default=None, 
                        help="Name of the HuggingFace Hub repository or the local file path for it")
    # Distributed Settings
    parser.add_argument("--num-gpus",
                        type=int,
                        default=None,
                        help='Number of GPUs used for training')
    parser.add_argument("--tensor-parallel-size", "-tp",
                        type=int,
                        default=None,
                        help='Tensor parallel degree (1 if not used)')
    parser.add_argument("--pipeline-parallel-size", "-pp",
                        type=int,
                        default=None,
                        help='Pipeline parallel degree (1 if not used)')
    parser.add_argument("--partition-activations", "-pa",
                        action="store_true",
                        help='Whether we use ZeRO-R to partition activation memory across tensor-parallel degree')
    parser.add_argument("--zero-stage", "-z",
                        type=int,
                        default=None,
                        choices=[0,1,2,3],
                        help='Stage of the ZeRO optimizer')
    parser.add_argument("--zero-allgather-bucket-size", "-zbs",
                        type=int,
                        default=None,
                        help='Size of allgather buckets used by ZeRO')
    parser.add_argument("--zero3-max-live-params", "-zmlp",
                        type=int,
                        default=None,
                        help='Maximum number of parameters ZeRO3 keeps in GPU memory')
    # Training settings
    parser.add_argument("--checkpoint-activations", "-ca",
                        action="store_true",
                        help='Whether Megatron-style activation checkpointing is being used')
    parser.add_argument("--batch-size-per-gpu", "-b",
                        type=int,
                        default=None,
                        help='Batch size per GPU')
    parser.add_argument("--sequence-length", "-s",
                        type=int,
                        default=None,
                        help='Sequence length used for training')
    parser.add_argument("--vocab-size", "-v",
                        type=int,
                        default=None,
                        help='How many tokens are in the embedding layer')
    # Model settings
    parser.add_argument("--hidden-size", "-hs",
                        type=int,
                        default=None,
                        help='Dimension of the model\'s hidden size')
    parser.add_argument("--num-attention-heads", "-a",
                        type=int,
                        default=None,
                        help='Number of attention heads used in model')
    parser.add_argument("--num-layers", "-l",
                        type=int,
                        default=None,
                        help='Number of transformer layers used in model')
    parser.add_argument("--ffn-expansion-factor", "-ff",
                        type=float,
                        default=None,
                        help='How much the MLP hidden size expands (can be non-integer like 3.5 for Llama models)')
    parser.add_argument("--num-mlp-linears", "-nl",
                        type=int,
                        default=None,
                        help='How many linear layers per MLP block')
    # Inference settings
    parser.add_argument("--infer",
                        action="store_true",
                        help="whether we're doing inference")
    parser.add_argument("--kv-size-ratio", "-kv",
                        type=float,
                        default=None,
                        help='Ratio of total query heads to key/value heads. 1.0 for MHA, 1/num_attention_heads for MQA.')
    parser.add_argument("--output-tokens", "-o",
                        type=int,
                        default=None,
                        help='Number of tokens to autoregressively generate.')
    # Precision settings
    parser.add_argument("--disable-mixed-precision",
                        action="store_false",
                        help='Disables mixed precision training',
                        dest='is_mixed_precision')
    parser.add_argument("--high-prec-bytes-per-val",
                        type=int,
                        default=None,
                        help='The high-precision bytes per value (parameter, optimizer state, etc) in mixed precision')
    parser.add_argument("--low-prec-bytes-per-val",
                        type=int,
                        default=None,
                        help='The low-precision bytes per value (parameter, optimizer state, etc) in mixed precision')
    parser.add_argument("--bytes-per-grad-ele",
                        type=int,
                        default=None,
                        help='The precision of gradient elements as bytes per value')
    parser.add_argument("--optimizer", "-opt",
                        type=str,
                        choices=["adamw", "muon", "sgd_momentum"],
                        default=None,
                        help='Which optimizer to estimate memory for (default adamw)')
    # MoE Settings
    parser.add_argument("--num-experts",
                        type=int,
                        default=None,
                        help='Number of experts')
    parser.add_argument("--expert-parallelism", "-ep",
                        type=int,
                        default=None,
                        help='How many ways are the experts sharded across ranks')
    # Miscellaneous memory (good for accounting for implementation-dependent fudge factors)
    parser.add_argument("--misc-mem-gib",
                        type=int,
                        default=None,
                        help='Miscellaneous memory overhead per GPU by DL framework(s), communication libraries, etc')

    return parser

DEFAULTS = {
    # Distributed Settings
    "num_gpus" : 1,
    "tensor_parallel_size" : 1,
    "pipeline_parallel_size" : 1,
    "partition_activations" : False,
    "zero_stage" : 1,
    "zero_allgather_bucket_size" : 5e8,
    "zero3_max_live_params" : 1e9,
    # Training Settings
    "checkpoint_activations" : False,
    "batch_size_per_gpu" : 1,
    "sequence_length" : 2048,
    "vocab_size" : 51200,
    # Model Settings
    "hidden_size" : 6144,
    "num_attention_heads" : 64,
    "num_layers" : 44,
    "ffn_expansion_factor" : 4.0,
    "num_mlp_linears": 2,
    # Inference Settings
    "infer" : False,
    "kv_size_ratio" : 1.0,
    "output_tokens" : 1,
    # Precision Settings
    "is_mixed_precision" : True,
    "high_prec_bytes_per_val" : 4,
    "low_prec_bytes_per_val" : 2,
    "bytes_per_grad_ele" : 4,
    "optimizer" : "adamw",
    # MoE Settings
    "num_experts" : 0,
    "expert_parallelism" : 1,
    # Miscellaneous Memory
    "misc_mem_gib" : 0
}

### End Argument Parsing ###

### Begin Memory Calculation ###

# Calculates the total memory necessary for model training or inference
def calc_mem(args):

    # set the hf_args if hf model is provided
    args = get_hf_model_args(args)

    dp_degree = args.num_gpus / (args.tensor_parallel_size * args.pipeline_parallel_size)

    # Compute total parameters from the config
    embed_params = 2 * args.vocab_size * args.hidden_size
    positional_params = args.hidden_size * args.sequence_length
    ln_params = 8 * args.hidden_size * args.num_layers + (2 * args.hidden_size)
    attention_params = int(2 * (1 + args.kv_size_ratio) * args.num_layers * args.hidden_size * args.hidden_size)
    mlp_params = args.num_mlp_linears * args.num_layers * args.hidden_size * args.ffn_expansion_factor * args.hidden_size
    total_params = embed_params + positional_params + ln_params + attention_params + mlp_params

    # --- MODEL MEMORY ---
    # 4 bytes in fp32, 2 bytes in fp16/bf16, 1 byte in fp8
    if args.is_mixed_precision:
        bytes_per_param = args.low_prec_bytes_per_val
    else:
        bytes_per_param = args.high_prec_bytes_per_val

    # Compute memory from param calculation and parallelism settings
    model_mem = total_params * bytes_per_param
    per_gpu_model_mem = model_mem
    if args.num_experts > 0:
        total_moe_params = embed_params + positional_params + ln_params + attention_params + (args.num_experts * mlp_params)
    # Split the model with 3D parallelism
    if args.num_experts == 0:
        per_gpu_model_mem = (total_params * bytes_per_param) / (args.tensor_parallel_size * args.pipeline_parallel_size)
    else:
        EP_total_params = embed_params + positional_params + ln_params + attention_params + ((args.num_experts/args.expert_parallelism) * mlp_params)
        per_gpu_model_mem = (EP_total_params * bytes_per_param) / (args.tensor_parallel_size * args.pipeline_parallel_size)
    # ZeRO stage 3 shards the model parameters across GPUs (plus the gradients and optimizer states)
    if args.zero_stage == 3:
        per_gpu_model_mem /= args.num_gpus

    # --- GRADIENT MEMORY ---
    # E.g. 4 bytes in fp32, 2 bytes in fp16/bf16, 1 byte in fp8
    # Gradient precision is sometimes configurable in training frameworks.
    # Since high batch size means many accumulations, higher precision grads may reduce grad overflow.
    bytes_per_grad_element = args.bytes_per_grad_ele

    if args.num_experts > 0:
        gradient_mem = EP_total_params * bytes_per_grad_element
    else:
        gradient_mem = total_params * bytes_per_grad_element
    per_gpu_gradient_mem = gradient_mem
    # ZeRO stage 2 shards the gradients across GPUs (plus the optimizer states)
    if args.zero_stage >= 2:
        per_gpu_gradient_mem /= args.num_gpus

    # --- OPTIMIZER MEMORY ---
    # For mixed-precision Adam/AdamW, the optimizer must store fp32 copies of the parameters, momentum, and variance (4 + 4 + 4 = 12 bytes per optimizer parameter)
    # Feel free to change the multiplier for your optimizer (examples include SGD w/ momentum (4 + 4 = 8) and 8-bit ADAM (2 + 2 + 2 = 6)
    opt_type = args.optimizer.lower()
    if opt_type == "sgd_momentum":
        opt_multiplier = 8                         # fp32 copy + m
    elif opt_type == "muon":
        opt_multiplier = 8                         # fp32 copy + m (https://kellerjordan.github.io/posts/muon/#runtime-analysis)
    else:  # Adamw default
        opt_multiplier =  12                       # fp32 copy + m + v (https://arxiv.org/abs/1910.02054)



    if args.num_experts > 0:
        optimizer_mem = EP_total_params * opt_multiplier
    else:
        optimizer_mem = total_params * opt_multiplier


    per_gpu_optimizer_mem = optimizer_mem
    # ZeRO stage 3 shards the optimizer states across GPUs
    if args.zero_stage >= 1:
        per_gpu_optimizer_mem /= args.num_gpus

    # --- COMMUNICATION MEMORY ---
    # Temporary GPU storage for communication buffers may become significant
    per_gpu_communication_mem = 0
    # The size of the communication buffer DeepSpeed uses to store ZeRO optimizer elements
    if args.zero_stage >= 1 and args.num_gpus > 1:
        per_gpu_communication_mem += args.zero_allgather_bucket_size * bytes_per_param
    # The number of parameters ZeRO-3 keeps alive in GPU memory at a time
    if args.zero_stage == 3 and args.num_gpus > 1:
        per_gpu_communication_mem += args.zero3_max_live_params * bytes_per_param

    # --- ACTIVATION MEMORY ---
    # Taken from Table 2 in https://arxiv.org/pdf/1910.02054.pdf and generalized to any precision (instead of just fp16 from the paper)
    # 3 cases: [training with activation checkpointing, training without activation checkpointing, inferencing]
    if not args.infer and args.checkpoint_activations:
        activation_mem = args.sequence_length * args.batch_size_per_gpu * args.hidden_size * args.num_layers * ((16 * args.low_prec_bytes_per_val + 2))
    elif not args.infer and not args.checkpoint_activations:
        activation_mem = args.sequence_length * args.batch_size_per_gpu * args.hidden_size * args.num_layers * ((16 * args.low_prec_bytes_per_val + 2) + (2 * args.low_prec_bytes_per_val + 1) * (args.num_attention_heads * args.sequence_length / args.hidden_size))
    # If using inference, assume just a single layer's activation memory at peak
    elif args.infer:
        activation_mem = args.sequence_length * args.batch_size_per_gpu * args.hidden_size * ((16 * args.low_prec_bytes_per_val + 2))
    per_gpu_activation_mem = activation_mem
    # DeepSpeed's ZeRO-R partitions activation memory across tensor-parallel GPUs
    if args.partition_activations:
        per_gpu_activation_mem = activation_mem / args.tensor_parallel_size

    # --- KV CACHE MEMORY (IF INFERENCE) ---
    if args.infer:
        # See https://kipp.ly/transformer-inference-arithmetic/ for details
        bytes_per_param = args.low_prec_bytes_per_val
        per_gpu_kv_cache_mem = bytes_per_param * args.hidden_size * args.num_layers * (args.sequence_length + args.output_tokens) * (args.batch_size_per_gpu)
        kv_cache_mem = args.num_gpus * per_gpu_kv_cache_mem

    gradient_mem_gib = gradient_mem / 1024**3
    activation_mem_gib = activation_mem / 1024**3
    model_mem_gib = model_mem / 1024**3
    optimizer_mem_gib = optimizer_mem / 1024**3

    per_gpu_gradient_mem_gib = per_gpu_gradient_mem / 1024**3
    per_gpu_activation_mem_gib = per_gpu_activation_mem / 1024**3
    per_gpu_model_mem_gib = per_gpu_model_mem / 1024**3
    per_gpu_optimizer_mem_gib = per_gpu_optimizer_mem / 1024**3
    per_gpu_communication_mem_gib = per_gpu_communication_mem / 1024**3
    

    # We include a "Miscellaneous Memory" per GPU term because we find some 3D-parallel frameworks add a constant memory overhead (~5GiB in our experiments with Megatron-DeepSpeed) that we cannot explain. If you know the source of this, add a comment!
    if args.infer:
        kv_cache_mem_gib = kv_cache_mem / 1024**3
        per_gpu_kv_cache_mem_gib = per_gpu_kv_cache_mem / 1024**3

    if args.infer:
        per_gpu_mem_gib = per_gpu_activation_mem_gib + per_gpu_kv_cache_mem_gib + per_gpu_model_mem_gib + args.misc_mem_gib
        single_replica_mem_gib = activation_mem_gib + kv_cache_mem_gib + model_mem_gib + args.misc_mem_gib * args.num_gpus
    else:
        per_gpu_mem_gib = per_gpu_activation_mem_gib + per_gpu_gradient_mem_gib + per_gpu_model_mem_gib + per_gpu_optimizer_mem_gib + per_gpu_communication_mem_gib + args.misc_mem_gib
        single_replica_mem_gib = activation_mem_gib + gradient_mem_gib + model_mem_gib + optimizer_mem_gib + args.misc_mem_gib * args.num_gpus

    # Print number of forward-pass parameters, and account for experts if using MoE
    print(f'Calculating memory with training configuration: {vars(args)}\n')
    print(f'Number of Parameters: {convert_params(total_params)}')
    if args.num_experts > 0:
        print(f'Total Number of MoE Parameters: {convert_params(total_moe_params)}')
    print()

    # Print per-GPU memory for each component
    print(f'*** Per-GPU Memory')
    print(f'Per-GPU Activation Memory: {per_gpu_activation_mem_gib:.2f} GiB')
    print(f'Per-GPU Model Memory: {per_gpu_model_mem_gib:.2f} GiB')
    if args.infer:
        print(f'Per-GPU KV Cache Memory: {per_gpu_kv_cache_mem_gib:.2f} GiB')
    else:
        print(f'Per-GPU Gradient Memory: {per_gpu_gradient_mem_gib:.2f} GiB')
        print(f'Per-GPU Optimizer Memory ({opt_type}): {per_gpu_optimizer_mem_gib:.2f} GiB')
        print(f'Per-GPU Communication Memory: {per_gpu_communication_mem_gib:.2f} GiB')
        print(f'Per-GPU Miscellaneous Memory: {args.misc_mem_gib:.2f} GiB')
    # Aggregate Per-GPU Memory
    if args.infer:
        print(f'\nPer-GPU Memory Required for Inference: {per_gpu_mem_gib:.2f} GiB')
    else:
        print(f'\nPer-GPU Memory Required for Training: {per_gpu_mem_gib:.2f} GiB')
    print()

    # Print total GPU memory required to store a complete model replica
    print(f'*** Total GPU Memory for a Single Model Replica')
    print(f'Total Activation Memory: {activation_mem_gib:.2f} GiB')
    print(f'Total Model Memory: {model_mem_gib:.2f} GiB')
    if args.infer:
        print(f'Total KV Cache Memory: {kv_cache_mem_gib:.2f} GiB')
    else:
        print(f'Total Gradient Memory: {gradient_mem_gib:.2f} GiB')
        print(f'Total Optimizer Memory ({opt_type}): {optimizer_mem_gib:.2f} GiB')
        print(f'Total Miscellaneous Memory: {args.num_gpus*args.misc_mem_gib:.2f} GiB')
    # Aggregate GPU memory
    if args.infer:
        print(f'\nTotal GPU Memory Required to Store a Complete Model Replica for Inference: {single_replica_mem_gib:.2f} GiB')
    else:
        print(f'\nTotal GPU Memory Required to Store a Complete Model Replica for Training: {single_replica_mem_gib:.2f} GiB')

### End Memory Calculation ###

if __name__ == "__main__":
    print('\nExample with pythia 6.9B: python calc_transformer_mem.py --num-layers=32 --sequence-length=2048 --num-attention-heads=32 --hidden-size=4096 --batch-size-per-gpu=8 --checkpoint-activations --zero-stage=1 --partition-activations --pipeline-parallel-size=1 --tensor-parallel-size=2 --num-gpus=128')
    print('Example with pythia 12B: python calc_transformer_mem.py --num-layers=36 --sequence-length=2048 --num-attention-heads=40 --hidden-size=5120 --batch-size-per-gpu=8 --checkpoint-activations --zero-stage=1 --partition-activations --pipeline-parallel-size=1 --tensor-parallel-size=4 --num-gpus=256')
    print('Example with default 20B: python calc_transformer_mem.py --num-layers=44 --sequence-length=2048 --num-attention-heads=64 --hidden-size=6144 --batch-size-per-gpu=1 --checkpoint-activations --zero-stage=1 --partition-activations --pipeline-parallel-size=1 --tensor-parallel-size=1 --num-gpus=1\n')
    print('Example with llama 7B: python calc_transformer_mem.py --num-layers=32 --sequence-length=4096 --num-attention-heads=32 --hidden-size=4096 --ffn-expansion-factor=3.5 --batch-size-per-gpu=1 --zero-stage=1 --pipeline-parallel-size=1 --tensor-parallel-size=1 --num-gpus=1')
    args = config_parser().parse_args()
    calc_mem(args)