# 推理优化

本文讨论了 LLM 推理中最紧迫的挑战，并提供了一些实际解决方案。

## 理解 LLM 推理

大多数流行的 decoder-only LLM（例如 GPT-3）都是在因果建模(causal modeling)目标上进行预训练的，本质上是下一个词的预测器。这些 LLM 将一系列 Token 作为输入，并自回归地生成后续 Token，直到满足停止条件（例如生成的 Token 数量限制或停止词列表）或生成特殊的 `<end>` Token表示生成结束。

该过程包括两个阶段：预填充阶段(prefill phase)和解码阶段(decode phase)。

请注意，Token 是模型处理的语言的最小单元。一个 Token 大约相当于四个英文字符。在输入模型之前，所有自然语言的输入都会被转换为 Token。

## 预填充阶段

在预填充阶段，LLM 处理输入 Token 以计算中间状态（键和值），这些中间状态用于生成“第一个”新 Token。每个新 Token 都依赖于所有先前的 Token，但由于完整的输入范围已知，在高层次上，这是一个高度并行化的矩阵-矩阵操作。它有效地饱和了 GPU 的利用率。

## 解码阶段

在解码阶段，LLM 自回归地逐个生成输出 Token，直到满足停止条件。每个顺序输出 Token 都需要了解所有先前迭代的输出状态（键和值）。与预填充阶段相比，这类似于一种矩阵-向量操作，但对于 GPU 的计算能力利用不足。数据（权重、键、值、激活值）从内存传输到 GPU 的速度主导了延迟，而不是计算本身的速度。换句话说，这是一种 memory-bound 的操作。

本文中介绍的许多推理挑战及相应解决方案都涉及到对解码阶段的优化：有效的注意力模块、有效管理键和值等。

不同的 LLM 可能使用不同的分词器，因此，直接比较它们之间的输出 Token 可能并不简单。当比较推理吞吐量时，即使两个 LLM 的每秒输出 Token 数相似，如果它们使用不同的分词器，它们可能并不等效。这是因为相应的 Token 可能代表不同数量的字符。

## 批处理(Batching)

改善 GPU 利用率并提高吞吐量的最简单方法是通过批处理。由于多个请求使用同一个模型，权重的内存成本得到了分摊。将较大的批次一次性传输到 GPU 进行处理，将更充分地利用可用的计算资源。

然而，批次大小只能增加到一定的限制，超过这个限制可能会导致内存溢出。要更好地理解为什么会出现这种情况，需要查看键-值（KV）缓存和 LLM 内存需求。

传统的批处理（也称为静态批处理）是次优的。这是因为在批处理中的每个请求中，LLM 可能会生成不同数量的完成 Token，进而导致它们具有不同的执行时间。结果是，批处理中的所有请求必须等待最长的请求完成，这可能会因生成长度的差异较大而加剧。有一些方法可以缓解这个问题，例如 in-flight batching，稍后将进行讨论。

## Key-value caching

解码阶段的一种常见优化是键-值（KV）缓存。解码阶段在每个时间步生成一个 Token，但每个 Token 都依赖于所有先前 Token 的键和值张量（包括预填充时计算的输入Token 的 KV 张量，以及计算到当前时间步的任何新 KV 张量）。

为了避免在每个时间步骤中为所有 Token 重新计算所有这些张量，可以将它们缓存在 GPU 内存中。在每次迭代中，当计算出新元素时，它们只需添加到运行缓存中，以便在下一次迭代中使用。在某些实现中，模型的每一层都有一个 KV 缓存。

![](./assets/nv_key-value-caching_.png)

## LLM 的显存要求

实际上，对于 GPU 上的 LLM 显存需求，主要的两个因素是模型权重和 KV 缓存。

- 模型权重：显存被模型参数占用。例如，一个具有 7B 参数的模型（如 Llama2 7B），以 16-bit 精度（FP16 或 BF16）加载，大约需要 `7B * sizeof(FP16) ~= 14GB` 的内存。

- KV 缓存：内存被 self-attention 张量的缓存占用，以避免冗余计算。
在批处理中，批中每个请求的KV缓存仍然必须单独分配，并且可能占用大量内存。下面的公式描述了适用于大多数常见LLM架构的KV缓存大小。

```
每个 Token 的 KV 缓存大小（以字节为单位）= 2 * (num_layers) * (num_heads * dim_head) * precision_in_bytes
```

第一个因子 2 代表 K 和 V 矩阵。通常，(num_heads * dim_head) 的值与 transformer 的 hidden_size（或模型的维度 d_model）相同。这些模型属性通常可以在模型卡或相关的配置文件中找到。

每个输入序列中的每个 Token 都需要这个内存大小，跨输入批次。假设使用半精度，KV 缓存的总大小由以下公式给出。

```
KV 缓存的总大小（以字节为单位）= (batch_size) * (sequence_length) * 2 * (num_layers) * (hidden_size) * sizeof(FP16)
```

例如，对于一个使用 16-bit 精度、批大小为 1 的 Llama2 7B 模型，

```
一个 Token 的 KV 缓存大小：1 * 1 * 2 * 32 * 4096 * 2 = 512KB
1024 个 Token 的 KV 缓存的大小将为 1 * 1024 * 2 * 32 * 4096 * 2 = 512MB
4096 个 Token 的 KV 缓存的大小将为 1 * 4096 * 2 * 32 * 4096 * 2 = 2048MB = 2GB。
```

有效地管理这个 KV 缓存是一项具有挑战性的任务。由于随着批大小和序列长度的线性增长，内存需求可能迅速增加。因此，它限制了可以提供的吞吐量，并对长上下文输入提出了挑战。这是本文中介绍的几项优化的动机。

## 参考文献
- [原文：mastering-llm-techniques-inference-optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)