# The Cookbook
Deep learning for dummies. All the practical details that go into working with real models.



## Table of Contents

### Calculations

For training/inference calculations (e.g. FLOPs, memory overhead, and parameter count)
- **[calc](./calc/)**

### Benchmarks

For benchmarks (e.g. communication)
- **[benchmarks](./benchmarks/)**


## Reading List and Similar Resources

[Transformers Math 101](https://blog.eleuther.ai/transformer-math/). A blog post from EleutherAI on training/inference memory estimations, parallelism, FLOP calculations, and deep learning datatypes

[LLM Visualizations](https://bbycroft.net/llm). Clear LLM visualizations and animations for basic transformer understanding.

[Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/). A breakdown on the memory overhead, FLOPs, and latency of transformer inference

[ML-Engineering Repository](https://github.com/stas00/ml-engineering). Containing community notes and practical details of everything deep learning training led by Stas Bekman

[LLM Finetuning Memory Requirements](https://blog.scottlogic.com/2023/11/24/llm-mem.html) by Alex Birch. A practical guide on the memory overhead of finetuning models.

[Annotated PyTorch Paper Implementations](https://nn.labml.ai/)

[Everything about Distributed Training and Efficient Finetuning](https://sumanthrh.com/post/distributed-and-efficient-finetuning/) by Sumanth R Hegde. High-level descriptions and links on parallelism and efficient finetuning.


## Minimal Repositories for Understanding

GPT Inference
- https://github.com/pytorch-labs/gpt-fast/tree/main

[RWKV](https://www.rwkv.com/)
- https://github.com/Hannibal046/nanoRWKV/tree/main

GPT Training
- https://github.com/karpathy/minGPT

Architecture-Specific Examples
- https://github.com/zphang/minimal-gpt-neox-20b
- https://github.com/zphang/minimal-llama
- https://github.com/zphang/minimal-opt

## Contributing

If you found a bug, typo, or would like to propose an improvement please don't hesitate to open an [Issue](https://github.com/EleutherAI/cookbook/issues) or contribute a [PR](https://github.com/EleutherAI/cookbook/pulls).
