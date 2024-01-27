import contextlib
from dataclasses import dataclass,asdict
import os
import torch

import megatron


@dataclass
class Arguments:
    precision: str = "fp16"
    #fp16 : bool = True
    #bf16 : bool = False
    apply_query_key_layer_scaling : bool = True
    attention_softmax_in_fp32 : bool = False
    scaled_masked_softmax_fusion : bool = True
    attention_dropout : float = 0.0
    #kv_channels : int = 128
    num_attention_heads : int = 8
    hidden_size : int = 1024
    rank : int = 0
    local_rank : int = 0
    distributed_backend : str = "nccl"
    world_size : int = 1
    model_parallel_size : int = 1
    pipe_parallel_size : int = 1
    global_num_gpus : int = 1
    virtual_pipeline_model_parallel_size = None
    pipeline_model_parallel_split_rank = None
    #no_async_tensor_model_parallel_allreduce : bool = True
    seq_length : int = 2048
    train_micro_batch_size_per_gpu : int = 2
    #train_batch_size : int = 2048
    #gradient_accumulation_steps: int = None
    use_cpu_initialization : bool = False
    params_dtype = torch.float16
    #ffn_hidden_size : int = 4096
    num_layers : int = 2
    bias_gelu_fusion : bool = True
    #openai_gelu : bool = False
    onnx_safe = None
    #apply_residual_connection_post_layernorm : bool = False
    #fp32_residual_connection : bool = False
    bias_dropout_fusion : bool = True
    layernorm_epsilon : float = 1e-5
    hidden_dropout : float = 0.0
    fp16_lm_cross_entropy : bool = False
    init_method_std : float = 0.02
    padded_vocab_size : int = 51200
    max_position_embeddings : int = 2048 #originally 1024
    activations_checkpoint_method = None
    checkpoint_num_layers : int = 1
    #distribute_checkpointed_activations : bool = False
    #no_persist_layer_norm : bool = False
    #DDP_impl : str = "local"
    #accumulate_allreduce_grads_in_fp32 : bool = False
    #use_contiguous_buffers_in_local_ddp : bool = True
    optimizer_type : str = "adam"
    lr : float = 0.00015
    weight_decay : float = 0.01
    #adam_beta1 : float = 0.9
    #adam_beta2 : float = 0.999
    #adam_eps : float = 1e-08
    loss_scale = None
    #initial_loss_scale : int = 4294967296
    #min_loss_scale : float = 1.0
    loss_scale_window : int = 1000
    hysteresis : int = 2
    clip_grad : float = 1.0
    #log_num_zeros_in_grad : bool = False
    train_iters : int = 20000
    lr_decay_iters : int = 20000
    lr_decay_style : str = "cosine"
    #train_batch_size : int = 512
    #lr_warmup_fraction : float = 0.01
    #min_lr : float = 1e-05
    use_checkpoint_lr_scheduler : bool = False
    override_lr_scheduler : bool = False
    load = None
    save : str = None
    seed : int = 1234
    #data_parallel_random_init : bool = False


def initialize_megatron(configuration):
    with open("/dev/null", 'w') as f:
        with contextlib.redirect_stdout(f):
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "6000"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            args = get_megatron_args(configuration, override_tensor_mp_size=True)
            #megatron.global_vars._GLOBAL_ARGS = args
            neox_args = megatron.NeoXArgs.from_dict(asdict(args))
            megatron.initialize._initialize_distributed(neox_args=neox_args)
            megatron.initialize._set_random_seed(neox_args.seed)
            #megatron.initialize._compile_dependencies()


def get_megatron_args(configuration, override_tensor_mp_size=False):
    (microbatch_size, hidden_size, (tensor_mp_size, pipeline_mp_size, dp_size), num_attention_heads,vocab_size,seq_length,train_batch_size) = configuration
    args = Arguments()
    args.params_dtype = torch.half
    if not override_tensor_mp_size:
        args.tensor_model_parallel_size = tensor_mp_size
    args.train_micro_batch_size_per_gpu = microbatch_size
    args.hidden_size = hidden_size
    args.ffn_hidden_size = 4 * args.hidden_size
    args.num_attention_heads = num_attention_heads
    args.kv_channels = args.hidden_size // args.num_attention_heads
    args.padded_vocab_size=vocab_size
    args.attention_config = [[["flash"], 0]]
    #megatron.global_vars._GLOBAL_ARGS = args
    neox_args = megatron.NeoXArgs.from_dict(asdict(args))
    return neox_args
