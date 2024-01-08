import time
import torch
import os
import numpy as np
import megatron_wrapper
import megatron
from megatron.model.word_embeddings import Embedding
import sys

arch = "a100"
# v_sweep

configurations=[]
for tensor_mp_size in [1]:
    for num_attention_heads in [128]:
        for v in range(51200-64,51200+64+1):
            for seq_length in [2048]:
                for batch_size in [4]:
                    for h in [2048]:
                        configurations.append((batch_size, h,
                                           (tensor_mp_size, 1, 1), num_attention_heads,v))


with open(f'results/transformer_data/{arch}/vocab_v_sweep.out', 'w') as sys.stdout:
    megatron_wrapper.initialize_megatron(configurations[0])
    for configuration in configurations:
        (microbatch_size, hidden_size,
        (tensor_mp_size, pipeline_mp_size, dp_size), num_attention_heads,vocab_size) = configuration
        args = megatron_wrapper.get_megatron_args(configuration)
        inp = torch.randint(vocab_size-1,(seq_length, batch_size)).to("cuda:0")
        init_method = megatron.model.init_functions.init_method_normal(args.init_method_std)
        embedding_layer = Embedding(args,h,vocab_size,seq_length,0.0,init_method=init_method,use_pos_emb=False)

        start_time=0
        for i in range(150):
            if i == 50:
                torch.cuda.synchronize()
                start_time = time.time()
            embedding_layer(inp,None)
            torch.cuda.synchronize()
        latency = time.time()-start_time
        print(f"{vocab_size} {latency}")

# h_sweep

configurations=[]
for tensor_mp_size in [1]:
    for num_attention_heads in [128]:
        for v in [51200]:
            for seq_length in [2048]:
                for batch_size in [4]:
                    for h in range(2048-64,2048+64):
                        configurations.append((batch_size, h,
                                           (tensor_mp_size, 1, 1), num_attention_heads,v))

with open(f'results/transformer_data/{arch}/vocab_h_sweep.out', 'w') as sys.stdout:
    megatron_wrapper.initialize_megatron(configurations[0])
    for configuration in configurations:
        (microbatch_size, hidden_size,
        (tensor_mp_size, pipeline_mp_size, dp_size), num_attention_heads,vocab_size) = configuration
        args = megatron_wrapper.get_megatron_args(configuration)
        inp = torch.randint(vocab_size-1,(seq_length, batch_size)).to("cuda:0")
        init_method = megatron.model.init_functions.init_method_normal(args.init_method_std)
        embedding_layer = Embedding(args,h,vocab_size,seq_length,0.0,init_method=init_method,use_pos_emb=False)

        start_time=0
        for i in range(150):
            if i == 50:
                torch.cuda.synchronize()
                start_time = time.time()
            embedding_layer(inp,None)
            torch.cuda.synchronize()
        latency = time.time()-start_time
        print(f"{vocab_size} {latency}")