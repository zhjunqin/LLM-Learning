# TensorRT-LLM 内存使用情况

原文：https://nvidia.github.io/TensorRT-LLM/memory.html

本文档总结了 TensorRT-LLM 的内存使用情况，并解决了用户报告的常见问题和疑问。

# 理解推理时 GPU 内存使用情况

在推理时，对于从 TensorRT-LLM 模型生成的 TRT 引擎，GPU 内存使用的主要贡献者有三个：权重、内部激活张量和输入输出（IO）张量。对于 IO 张量，主要的内存占用来自 KV cache 张量。

## 权重大小
权重大小取决于模型大小、所选权重精度和并行化策略。使用较低精度，如 INT8 或 FP8，可以减小权重大小。当使用张量并行或流水线并行时，每个 rank 只存储权重的一部分。例如，在使用 8 路张量并行或 8 阶段流水线并行时，每个 rank 通常只使用模型权重的 1/8。

## 激活大小
TensorRT 可以通过基于实时分析和张量大小重用不同张量的内存来优化内存使用。为了避免运行时出现内存不足错误，并减少切换优化配置文件和改变形状的运行时成本，TensorRT 在构建时预先计算激活张量的内存需求。内存需求是基于优化后的 TensorRT 图计算的，一个配置文件的内存使用是通过使用最大张量形状来计算的，一个引擎的内存需求是通过不同配置文件之间的最大大小来计算的。影响 TensorRT 返回的激活大小的外部和内部因素包括网络结构、内核融合、操作调度等。一旦 TensorRT engine 构建完成，可以通过 API trt.ICudaEngine.device_memory_size 查询 engine 的激活内存大小。

实际上，对于给定的模型、指定的精度和并行化策略，可以通过调整最大批量大小、最大输入长度、最大束宽、最大 tokens 数、padding 移除开关、上下文 FMHA 开关等来调整激活内存使用。以下是这些值如何影响内存的一些解释：

1. 减少构建时最大输入 token

大多数 transformer 网络内的张量与输入 token 数呈线性关系，因此激活大小将接近最大输入 token 数 * 某个常数因子，这个常数因子取决于网络结构和 TRT 内部优化。最大输入 token 数来自构建时参数，可以通过更改 prepare_inputs 函数提供的参数（如 GPTLMHeadModel.prepare_inputs）来影响内存使用，或者可以更改 trtllm-build 命令的命令行选项。

当使用填充张量格式时，最大输入 token 数等于 `max_batch_size*max_input_len`，因此减少 max_batch_size 和 max_input_len 可以几乎线性地减少激活内存大小。当使用 padded tensors 格式并指定 max_num_tokens 时，减少其值也会减少激活内存大小。如果未指定 max_num_tokens，则最大输入 token 数将被推导为 `max_batch_size*max_input_len`。

建议使用 padded tensors 格式，因为它既节省内存又节省计算。束宽将在将张量范围传递到 TensorRT 时折叠到批量大小维度中，因此减少 max_beam_width 也可以减少内存使用。

2. 打开 context FMHA

当使用 GPT 注意力插件时，打开插件的 context_fmha_type 将显著减少内存占用。有关详细信息，请参见上下文阶段。当 context_fmha_type 设置为禁用时，插件的工作区大小将与序列长度成二次方关系。

3. 张量并行和流水线并行

TensorRT 将尽可能在层之间重用内存，例如，在一个 transformer 网络中给定 N 个解码器块，TRT 不会为每个块分配 N 份激活内存的副本，因为第 1 块的张量内存可以在执行后释放，内存可以重用于后续块，只需要 1 块的内存。

当使用张量并行时，一些张量被分成较小的块，每个 rank 只持有张量的一个块，每个 rank 的激活内存大小将小于在单个 GPU 上执行网络时的大小。当使用流水线并行时，每个 rank 执行几个解码器块，所有张量都是全尺寸张量，因此激活内存大小等于 1 块的内存大小。因此，在所有其他参数相同的情况下，张量并行通常比流水线并行具有更高的内存效率。

# KV 缓存张量

## Python 运行时
Python 运行时根据 GenerationSession.setup 函数的参数分配 KV 缓存张量，KV 缓存大小与批量大小和 `max_context_length+max_new_tokens` 成线性关系。注意：这在未来可能会改变，因为 C++ 运行时的 Python bindings 可能会在未来取代当前的 Python 运行时。C++ 运行时的 Python bindings 表现得像 C++ 运行时。

## C++ 运行时

### 启用分页 KV 缓存时

TensorRT-LLM 运行时在初始化时为配置的块数预先分配 KV cache 张量，并在运行时分配它们。创建 GptSession 时，KV cache 张量基于 KVCacheConfig 对象分配。如果既未指定 maxTokens 也未指定 freeGpuMemoryFraction，则 KV 缓存默认分配剩余免费 GPU 内存的 85%。如果指定了任一值，则使用指定值来计算 KV 缓存内存大小。如果两者都指定，则首先使用 freeGpuMemoryFraction 计算 KV 缓存中的 token 数，然后使用这个计算出的 token 数和 maxTokens 之间的最小值。

在 in-flight batching 中，只要有足够的 KV cache 空间，调度器就可以自动调度请求（确切行为取决于调度器策略）。如果在 GptSession 中使用 paged KV cache 而没有 in-flight batching，如果分页 paged KV cache 不足以容纳整个批次，TensorRT-LLM 可能会报告 OOM 错误，消息为“无法分配新块。没有空闲块”。

### 禁用分页 KV 缓存时

C++ 运行时为每层分配 KV 缓存张量，形状为 [batch size, 2, heads, max seq length, hidden dimension per head]，其中 max seq length 由创建 GptSession 时的 GptSession::Config::maxSequenceLength 指定。

# 内存池
TensorRT-LLM C++ 运行时使用 stream-ordered 内存分配器来分配和释放缓冲区，参见 BufferManager::initMemoryPool，它使用 CUDA 驱动程序管理的默认内存池。当 GptSession 对象被销毁时，内存将返回到内存池，并可被下一个 GptSession 对象实例重用。如果需要为其他内存分配释放内存，内存将从池中释放。然而，即使内存返回到 CUDA 驱动程序的内存池后，nvidia-smi 仍可能显示高内存占用。这不应该是一个问题，这是预期的行为。可以通过 BufferManager::memoryPoolReserved() 和 BufferManager::memoryPoolFree() 分别检查池中保留和空闲的内存量。

# 已知问题

当使用 2 个优化配置文件时，如果底层 kernel 选择需要不同优化配置文件的不同权重布局，权重内存大小在某些情况下可能会翻倍。这个问题将在未来的版本中修复。

当使用 FP8 GEMM 时，激活内存可能比理论优化的内存大小更大，这将在未来的版本中得到增强。

# 常见问题解答

1. 即使在运行时使用了小批量大小和序列长度，为什么内存大小仍然很大？

如上所述，激活内存大小是基于 TensorRT 引擎构建时的最大张量形状计算的，尝试减少引擎构建时的参数，如 max_num_token, max_batch_size, max_input_len，详见激活大小。

2. 为什么可以生成 engine，但推理时会运行出内存（OOM）？

在 engine 构建时，TensorRT 会逐层调整 kernel 选择，它不一定分配运行整个引擎所需的所有内存。如果运行单个层所需的激活张量很小，而运行 engine 所需的 I/O 张量（如 KV 缓存）大小很大，构建会成功，因为它可能不需要分配大的 I/O 张量，运行时可能因分配大的 I/O 张量而失败并出现 OOM 错误。

TensorRT-LLM 提供了一个 check_gpt_mem_usage 实用程序函数，用于检查给定 engine 的内存大小上限，以及相关的批量大小、I/O 序列长度等，当上限检查超过 GPU 物理内存大小时，将打印警告消息。

3. 如何调试 TensorRT-LLM 的内存使用？

当使用 verbose 日志级别时，TensorRT 和 TensorRT-LLM 将打印有关内存使用详细信息的消息。显示 “Total Weights Memory” 的行指示权重内存大小，而 “Total Activation Memory” 的行指示激活内存大小。

通常，权重内存大小接近 TensorRT engine 大小，因为 engine 中的大部分内容来自 LLM 网络的权重。

4. 对于流水线并行，构建时的最大 batch size 是 micro batch size 的限制吗？

是的，在流水线并行模式下，TensorRT-LLM 运行时将请求批次分割成 micro batches，并依次将这些微批次排队到 TRT engine 中。构建时的 max_batch_size 意味着每个 engine 入队调用的批量大小应小于它。在分割成微批次之前的总批量大小可以大于构建时的最大批量大小。

例如，如果您有 4 阶段的流水线并行，并且打算在一个生成调用中使用微批量大小 2 运行 16 个微批次（总批量大小 32）。您只需在构建时将 max_batch_size 设置为 2，而不是 32。