# 分布式训练并行化

简要概述各个并行化训练技术：

- DataParallel（DP）：相同的模型权重被复制多次，并且每个被提供数据的一个切片。处理是并行进行的，并且在每个训练步骤结束时进行同步。

- TensorParallel（TP）-每个张量被分割成多个块，因此每个张量的分片都驻留在其指定的GPU上，而不是整个张量都驻留在单个GPU上。在处理过程中，每个分片在不同的GPU上分别并行处理，并在步骤结束时进行同步。这可以称为水平并行，因为分割发生在水平层面上。

- PipelineParallel（PP）-模型在多个GPU上垂直（层级）分割，因此模型的一个或多个层仅放置在单个GPU上。每个GPU并行处理管道的不同阶段，并在一小批数据上工作。

Zero Redundancy Optimizer（ZeRO）-也执行张量的分片，与TP有些相似，但整个张量在前向或后向计算时会被重构，因此不需要修改模型。它还支持各种卸载技术，以弥补有限的GPU内存。

## 数据并行 (Data Parallelism)

## 张量并行 (Tensor Parallelism)

## 流水线并行 (Pipeline Parallelism)

## 



## 参考文献
- https://huggingface.co/docs/transformers/v4.17.0/en/parallelism