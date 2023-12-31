# The Cookbook
Deep learning for dummies, by Quentin Anthony, Hailey Schoelkopf, and Stella Biderman

All the practical details and utilities that go into working with real models! If you're just getting started, we recommend jumping ahead to [Basics](#basics) for some introductory resources on transformers.

## Table of Contents

- [The Cookbook](#the-cookbook)
  * [Calculations](#calculations)
  * [Benchmarks](#benchmarks)
  * [Reading List](#reading-list)
    + [Basics](#basics)
    + [How to do LLM Calculations](#how-to-do-llm-calculations)
    + [Distributed Deep Learning](#distributed-deep-learning)
    + [Best Practices](#best-practices)
  * [Minimal Repositories for Educational Purposes](#minimal-repositories-for-educational-purposes)
  * [Contributing](#contributing)

## Calculations

For training/inference calculations (e.g. FLOPs, memory overhead, and parameter count)
- **[calc](./calc/)**

Useful external calculators include

[Cerebras Model Lab](https://www.cerebras.net/model-lab/). User-friendly tool to apply Chinchilla scaling laws.

[Transformer Training and Inference VRAM Estimator](https://vram.asmirnov.xyz/) by Alexander Smirnov. A user-friendly tool to estimate VRAM overhead.

## Benchmarks

For benchmarks (e.g. communication)
- **[benchmarks](./benchmarks/)**

## Reading List

### Basics

[LLM Visualizations](https://bbycroft.net/llm). Clear LLM visualizations and animations for basic transformer understanding.

[Annotated PyTorch Paper Implementations](https://nn.labml.ai/)

[Jay Alammar's blog](https://jalammar.github.io/blog) contains many blog posts pitched to be accessible to a wide range of backgrounds. We recommend his posts [the Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), and [the Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) in particular.

[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) by Sasha Rush, Austin Huang, Suraj Subramanian, Jonathan Sum, Khalid Almubarak, and Stella Biderman. A walk through of the seminal paper "Attention is All You Need" along with in-line implementations in PyTorch.

### How to do LLM Calculations

[Transformers Math 101](https://blog.eleuther.ai/transformer-math/). A blog post from EleutherAI on training/inference memory estimations, parallelism, FLOP calculations, and deep learning datatypes.

[Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/). A breakdown on the memory overhead, FLOPs, and latency of transformer inference

[LLM Finetuning Memory Requirements](https://blog.scottlogic.com/2023/11/24/llm-mem.html) by Alex Birch. A practical guide on the memory overhead of finetuning models.

### Distributed Deep Learning

[Everything about Distributed Training and Efficient Finetuning](https://sumanthrh.com/post/distributed-and-efficient-finetuning/) by Sumanth R Hegde. High-level descriptions and links on parallelism and efficient finetuning.

[Efficient Training on Multiple GPUs](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_many) by Hugging Face. Contains a detailed walk-through of model, tensor, and data parallelism along with the ZeRO optimizer.

### Best Practices

[ML-Engineering Repository](https://github.com/stas00/ml-engineering). Containing community notes and practical details of everything deep learning training led by Stas Bekman

[Common HParam Settings](https://docs.google.com/spreadsheets/d/14vbBbuRMEHoqeuMHkTfw3uiZVmyXNuoSp8s-aHvfvZk/edit?usp=sharing) by Stella Biderman. Records common settings for model training hyperparameters and her current recommendations for training new models.

## Minimal Repositories for Educational Purposes

Large language models are frequently trained using very complex codebases due to the need to optimize things to work at scale and support a wide variety of configurable options. This can make them less useful pedagogical tools, so some people have developed striped-down so-called "Minimal Implementations" that are sufficient for smaller scale work and more pedagogically useful.

GPT Inference
- https://github.com/pytorch-labs/gpt-fast/tree/main

GPT Training
- https://github.com/karpathy/minGPT

Architecture-Specific Examples
- https://github.com/zphang/minimal-gpt-neox-20b
- https://github.com/zphang/minimal-llama
- https://github.com/zphang/minimal-opt

[RWKV](https://www.rwkv.com/)
- https://github.com/Hannibal046/nanoRWKV/tree/main


## Contributing

If you found a bug, typo, or would like to propose an improvement please don't hesitate to open an [Issue](https://github.com/EleutherAI/cookbook/issues) or contribute a [PR](https://github.com/EleutherAI/cookbook/pulls).
