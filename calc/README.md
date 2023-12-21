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
usage: calc_transformer_flops.py [-h] [--vocab-size VOCAB_SIZE] [--hidden-size HIDDEN_SIZE] [--sequence-length SEQUENCE_LENGTH] [--num-layers NUM_LAYERS] [--moe] [--num-experts NUM_EXPERTS] [--expert-interval EXPERT_INTERVAL]
                                 [--topk TOPK] [--batch-size BATCH_SIZE] [--tokens TOKENS] [--no-checkpoint-activations]

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
Example with pythia 6.9B: python transformer_mem.py --num-layers=32 --sequence-length=2048 --num-attention-heads=32 --hidden-size=4096 --batch-size-per-gpu=8 --checkpoint-activations --zero-stage=1 --partition-activations --pipeline-parallel-size=1 --tensor-parallel-size=2 --num-gpus=128 --params=6900000000
Example with pythia 12B: python transformer_mem.py --num-layers=36 --sequence-length=2048 --num-attention-heads=40 --hidden-size=5120 --batch-size-per-gpu=8 --checkpoint-activations --zero-stage=1 --partition-activations --pipeline-parallel-size=1 --tensor-parallel-size=4 --num-gpus=256 --params=11849420800
Example with default 20B: python transformer_mem.py --num-layers=44 --sequence-length=2048 --num-attention-heads=64 --hidden-size=6144 --batch-size-per-gpu=1 --checkpoint-activations --zero-stage=1 --partition-activations --pipeline-parallel-size=1 --tensor-parallel-size=1 --num-gpus=1 --params=20000000000

usage: calc_transformer_mem.py [-h] [--params PARAMS] [--num-gpus NUM_GPUS] [--tensor-parallel-size TENSOR_PARALLEL_SIZE] [--pipeline-parallel-size PIPELINE_PARALLEL_SIZE] [--partition-activations] [--zero-stage {0,1,2,3}]
                               [--checkpoint-activations] [--batch-size-per-gpu BATCH_SIZE_PER_GPU] [--hidden-size HIDDEN_SIZE] [--num-attention-heads NUM_ATTENTION_HEADS] [--sequence-length SEQUENCE_LENGTH] [--num-layers NUM_LAYERS]
                               [--fp32-model] [--fp32-grads] [--zero-allgather-bucket-size ZERO_ALLGATHER_BUCKET_SIZE] [--zero3-max-live-params ZERO3_MAX_LIVE_PARAMS] [--misc-mem-gb MISC_MEM_GB] [--num-experts NUM_EXPERTS]
                               [--ffn-expansion-factor FFN_EXPANSION_FACTOR] [--expert-parallelism EXPERT_PARALLELISM] [--vocab-size VOCAB_SIZE]

options:
  -h, --help            show this help message and exit
  --params PARAMS, -p PARAMS
                        Number of Parameters
  --num-gpus NUM_GPUS   Number of GPUs used for training
  --tensor-parallel-size TENSOR_PARALLEL_SIZE, -tp TENSOR_PARALLEL_SIZE
                        Tensor parallel degree (1 if not used)
  --pipeline-parallel-size PIPELINE_PARALLEL_SIZE, -pp PIPELINE_PARALLEL_SIZE
                        Pipeline parallel degree (1 if not used)
  --partition-activations, -pa
                        Whether we use ZeRO-R to partition activation memory across tensor-parallel degree
  --zero-stage {0,1,2,3}, -z {0,1,2,3}
                        Stage of the ZeRO optimizer
  --checkpoint-activations, -ca
                        Whether Megatron-style activation checkpointing is being used
  --batch-size-per-gpu BATCH_SIZE_PER_GPU, -b BATCH_SIZE_PER_GPU
                        Batch size per GPU
  --hidden-size HIDDEN_SIZE, -hs HIDDEN_SIZE
                        Dimension of the model's hidden size
  --num-attention-heads NUM_ATTENTION_HEADS, -a NUM_ATTENTION_HEADS
                        Number of attention heads used in model
  --sequence-length SEQUENCE_LENGTH, -s SEQUENCE_LENGTH
                        Sequence length used for training
  --num-layers NUM_LAYERS, -l NUM_LAYERS
                        Number of transformer layers used in model
  --fp32-model          Whether model is stored in fp32
  --fp32-grads          Whether grads are stored in fp32
  --zero-allgather-bucket-size ZERO_ALLGATHER_BUCKET_SIZE, -zbs ZERO_ALLGATHER_BUCKET_SIZE
                        Size of allgather buckets used by ZeRO
  --zero3-max-live-params ZERO3_MAX_LIVE_PARAMS, -zmlp ZERO3_MAX_LIVE_PARAMS
                        Maximum number of parameters ZeRO3 keeps in GPU memory
  --misc-mem-gb MISC_MEM_GB
                        Miscellaneous memory overhead by DL framework(s), communication libraries, etc
  --num-experts NUM_EXPERTS
                        Number of experts
  --ffn-expansion-factor FFN_EXPANSION_FACTOR, -ff FFN_EXPANSION_FACTOR
                        How much the MLP hidden size expands
  --expert-parallelism EXPERT_PARALLELISM, -ep EXPERT_PARALLELISM
                        How many ways are the experts sharded across ranks
  --vocab-size VOCAB_SIZE, -v VOCAB_SIZE
                        How many ways are the experts sharded across ranks
```


### Notes

Our scripts largely assume a standard transformer architecture as in GPT-NeoX or GPT-3, with parameter-free positional embeddings such as RoPE. Certain architectural choices may affect parameter counts, FLOPs, or memory overhead, such as positional embedding, multi-query attention (MQA), or other changes. These scripts should hold for models trained with SwiGLU activation functions such as Llama. 
