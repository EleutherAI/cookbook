# Calculations

Before training or inference even begins, common practical questions about potential models must be answered such as:
1. How many parameters are we targeting? How should those parameters be allocated within the model?
2. How many FLOPs does the model from step 1 take to train on *t* tokens? How about inference?
3. How much memory does the model from step 1 take to train/infer on *d* devices? What memory-saving strategies (e.g. parallelism, quantization, etc) are necessary to fit the model on device memory?


## Running Scripts

Currently, scripts are entirely self-contained. This is for the dual purpose of:
1. Making them easily shared and untied to this repository
2. Clarity on which script arguments are relevant to the contained calculation (e.g. number of attention heads affects memory overhead but not FLOPs or params. Abstracting args to a common utils location would blur this for newcomers)


### Calculating FLOPs

`calc_transformer_flops.py` calculates the number of theoretical FLOPs required to train a model on *t* tokens. See [Transformers Math 101](https://blog.eleuther.ai/transformer-math/) for more details on how FLOPs are calculated. Other good resources that we consulted are [the Chinchilla Paper](https://arxiv.org/abs/2203.15556) and [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://people.eecs.berkeley.edu/~matei/papers/2021/sc_megatron_lm.pdf).

```
Example with Fairseq-MoE 15B: python calc_transformer_flops.py -l 12 -hs 768 --moe -e 512
Example with GPT-3 175B: python calc_transformer_flops.py -l 96 -hs 12288
usage: calc_transformer_flops.py [-h] [--vocab-size VOCAB_SIZE] [--hidden-size HIDDEN_SIZE] [--sequence-length SEQUENCE_LENGTH] [--num-layers NUM_LAYERS] [--kv-size-ratio KV_SIZE_RATIO] [--moe] [--num-experts NUM_EXPERTS] [--expert-interval EXPERT_INTERVAL] [--topk TOPK] [--swiglu] [--batch-size BATCH_SIZE] [--tokens TOKENS] [--no-checkpoint-activations]

options:
  -h, --help            show this help message and exit
  --vocab-size VOCAB_SIZE, -v VOCAB_SIZE
                        Size of the vocab
  --hidden-size HIDDEN_SIZE, -hs HIDDEN_SIZE
                        Dimension of the model's hidden size
  --sequence-length SEQUENCE_LENGTH, -s SEQUENCE_LENGTH
                        Sequence length used for training
  --num-layers NUM_LAYERS, -l NUM_LAYERS
                        Number of transformer layers used in model
  --kv-size-ratio KV_SIZE_RATIO, -kv KV_SIZE_RATIO
                        Ratio of kv heads to query heads used in model. 1.0 for MHA
  --moe                 Whether our model is MoE
  --num-experts NUM_EXPERTS, -e NUM_EXPERTS
                        Number of experts for MoE
  --expert-interval EXPERT_INTERVAL, -ei EXPERT_INTERVAL
                        Expert interval for MoE
  --topk TOPK, -t TOPK  Top k routing for MoE
  --swiglu              Use swiglu MLP. If set, ffn-hidden-size is defined as the inner dimension of each of the three MLP weights.
  --batch-size BATCH_SIZE, -b BATCH_SIZE
                        Global batch size in units of samples
  --tokens TOKENS       Number of tokens you are training over
  --no-checkpoint-activations, -ca
                        Whether Megatron-style activation checkpointing is being used
```


### Calculating Parameters

`calc_transformer_params.py` calculates the number of parameters present in a given model based on its hyperparams. Such calculations are important to determine memory overheads, FLOPs, or to determine the size of an unknown transformer model. We also found the following resources helpful: [How does GPT-3 spend its 175B parameters?](https://www.lesswrong.com/posts/3duR8CrvcHywrnhLo/how-does-gpt-3-spend-its-175b-parameters) and [LLM Parameter Counting](https://kipp.ly/transformer-param-count/).

```
Example with Fairseq-MoE 15B: python calc_transformer_params.py -l 12 -hs 768 --moe -e 512
Example with GPT-3 175B: python calc_transformer_params.py -l 96 -hs 12288
usage: calc_transformer_params.py [-h] [--vocab-size VOCAB_SIZE] [--hidden-size HIDDEN_SIZE] [--sequence-length SEQUENCE_LENGTH] [--num-layers NUM_LAYERS] [--moe] [--num-experts NUM_EXPERTS] [--expert-interval EXPERT_INTERVAL]
                                  [--topk TOPK] [--ffn-expansion-factor FFN_EXPANSION_FACTOR]

options:
  -h, --help            show this help message and exit
  --vocab-size VOCAB_SIZE, -v VOCAB_SIZE
                        Size of the vocab
  --hidden-size HIDDEN_SIZE, -hs HIDDEN_SIZE
                        Dimension of the model's hidden size
  --sequence-length SEQUENCE_LENGTH, -s SEQUENCE_LENGTH
                        Sequence length used for training
  --num-layers NUM_LAYERS, -l NUM_LAYERS
                        Number of transformer layers used in model
  --moe                 Whether our model is MoE
  --num-experts NUM_EXPERTS, -e NUM_EXPERTS
                        Number of experts for MoE
  --expert-interval EXPERT_INTERVAL, -ei EXPERT_INTERVAL
                        Expert interval for MoE
  --topk TOPK, -t TOPK  Top k routing for MoE
  --ffn-expansion-factor FFN_EXPANSION_FACTOR, -ff FFN_EXPANSION_FACTOR
                        How much the MLP hidden size expands
```


### Calculating Memory Overhead

`calc_transformer_mem.py` calculates the amount of device memory required to train or infer a model. See [Transformers Math 101](https://blog.eleuther.ai/transformer-math/) for more details on how memory overhead is calculated. Take this estimation with a grain of salt, because every implementation is different and these calculations were written to match the GPT-NeoX library as close as possible. Even for other training and inference libraries, however, we expect our script to give approximate memory estimations within acceptable error. (Please see [LLM finetuning memory requirements](https://blog.scottlogic.com/2023/11/24/llm-mem.html) for a treatment of how specific memory costs may vary framework-to-framework). Other good resources that we consulted are [the ZeRO Paper](https://arxiv.org/abs/1910.02054) and [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/pdf/2205.05198.pdf).

```
Example with pythia 6.9B: python calc_transformer_mem.py --num-layers=32 --sequence-length=2048 --num-attention-heads=32 --hidden-size=4096 --batch-size-per-gpu=8 --checkpoint-activations --zero-stage=1 --partition-activations --pipeline-parallel-size=1 --tensor-parallel-size=2 --num-gpus=128
Example with pythia 12B: python calc_transformer_mem.py --num-layers=36 --sequence-length=2048 --num-attention-heads=40 --hidden-size=5120 --batch-size-per-gpu=8 --checkpoint-activations --zero-stage=1 --partition-activations --pipeline-parallel-size=1 --tensor-parallel-size=4 --num-gpus=256
Example with default 20B: python calc_transformer_mem.py --num-layers=44 --sequence-length=2048 --num-attention-heads=64 --hidden-size=6144 --batch-size-per-gpu=1 --checkpoint-activations --zero-stage=1 --partition-activations --pipeline-parallel-size=1 --tensor-parallel-size=1 --num-gpus=1

usage: calc_transformer_mem.py [-h] [--num-gpus NUM_GPUS] [--tensor-parallel-size TENSOR_PARALLEL_SIZE] [--pipeline-parallel-size PIPELINE_PARALLEL_SIZE] [--partition-activations] [--zero-stage {0,1,2,3}] [--zero-allgather-bucket-size ZERO_ALLGATHER_BUCKET_SIZE] [--zero3-max-live-params ZERO3_MAX_LIVE_PARAMS] [--checkpoint-activations] [--batch-size-per-gpu BATCH_SIZE_PER_GPU] [--sequence-length SEQUENCE_LENGTH] [--vocab-size VOCAB_SIZE] [--hidden-size HIDDEN_SIZE] [--num-attention-heads NUM_ATTENTION_HEADS]
                               [--num-layers NUM_LAYERS] [--ffn-expansion-factor FFN_EXPANSION_FACTOR] [--infer] [--kv-size-ratio KV_SIZE_RATIO] [--disable-mixed-precision] [--high-prec-bytes-per-val HIGH_PREC_BYTES_PER_VAL] [--low-prec-bytes-per-val LOW_PREC_BYTES_PER_VAL] [--bytes-per-grad-ele BYTES_PER_GRAD_ELE] [--num-experts NUM_EXPERTS] [--expert-parallelism EXPERT_PARALLELISM] [--misc-mem-gib MISC_MEM_GIB]

options:
  -h, --help            show this help message and exit
  --num-gpus NUM_GPUS   Number of GPUs used for training
  --tensor-parallel-size TENSOR_PARALLEL_SIZE, -tp TENSOR_PARALLEL_SIZE
                        Tensor parallel degree (1 if not used)
  --pipeline-parallel-size PIPELINE_PARALLEL_SIZE, -pp PIPELINE_PARALLEL_SIZE
                        Pipeline parallel degree (1 if not used)
  --partition-activations, -pa
                        Whether we use ZeRO-R to partition activation memory across tensor-parallel degree
  --zero-stage {0,1,2,3}, -z {0,1,2,3}
                        Stage of the ZeRO optimizer
  --zero-allgather-bucket-size ZERO_ALLGATHER_BUCKET_SIZE, -zbs ZERO_ALLGATHER_BUCKET_SIZE
                        Size of allgather buckets used by ZeRO
  --zero3-max-live-params ZERO3_MAX_LIVE_PARAMS, -zmlp ZERO3_MAX_LIVE_PARAMS
                        Maximum number of parameters ZeRO3 keeps in GPU memory
  --checkpoint-activations, -ca
                        Whether Megatron-style activation checkpointing is being used
  --batch-size-per-gpu BATCH_SIZE_PER_GPU, -b BATCH_SIZE_PER_GPU
                        Batch size per GPU
  --sequence-length SEQUENCE_LENGTH, -s SEQUENCE_LENGTH
                        Sequence length used for training
  --vocab-size VOCAB_SIZE, -v VOCAB_SIZE
                        How many tokens are in the embedding layer
  --hidden-size HIDDEN_SIZE, -hs HIDDEN_SIZE
                        Dimension of the model's hidden size
  --num-attention-heads NUM_ATTENTION_HEADS, -a NUM_ATTENTION_HEADS
                        Number of attention heads used in model
  --num-layers NUM_LAYERS, -l NUM_LAYERS
                        Number of transformer layers used in model
  --ffn-expansion-factor FFN_EXPANSION_FACTOR, -ff FFN_EXPANSION_FACTOR
                        How much the MLP hidden size expands
  --infer               whether we're doing inference
  --kv-size-ratio KV_SIZE_RATIO, -kv KV_SIZE_RATIO
                        Ratio of total query heads to key/value heads. 1.0 for MHA, 1/num_attention_heads for MQA.
  --disable-mixed-precision
                        Disables mixed precision training
  --high-prec-bytes-per-val HIGH_PREC_BYTES_PER_VAL
                        The high-precision bytes per value (parameter, optimizer state, etc) in mixed precision
  --low-prec-bytes-per-val LOW_PREC_BYTES_PER_VAL
                        The low-precision bytes per value (parameter, optimizer state, etc) in mixed precision
  --bytes-per-grad-ele BYTES_PER_GRAD_ELE
                        The precision of gradient elements as bytes per value
  --num-experts NUM_EXPERTS
                        Number of experts
  --expert-parallelism EXPERT_PARALLELISM, -ep EXPERT_PARALLELISM
                        How many ways are the experts sharded across ranks
  --misc-mem-gib MISC_MEM_GIB
                        Miscellaneous memory overhead per GPU by DL framework(s), communication libraries, etc
```


### Notes

Our scripts largely assume a standard transformer architecture as in GPT-NeoX or GPT-3, with parameter-free positional embeddings such as RoPE. Certain architectural choices may affect parameter counts, FLOPs, or memory overhead, such as positional embedding, multi-query attention (MQA), or other changes. These scripts should hold for models trained with SwiGLU activation functions such as Llama. 
