# Triton TRT-LLM backend

## Clone 代码

Clone tensorrtllm_backend 和 TensorRT-LLM。

```
git clone -b v0.8.0 https://github.com/triton-inference-server/tensorrtllm_backend.git

git submodule update --init --recursive

git lfs install
git lfs pull
```
TensorRT-LLM 作为子模块在 tensorrtllm_backend 子目录中。

## 编译 engine

参考[文档](./run_trt_llm.md)编译生成 TRT-LLM engine。

```
# Build a single-GPU float16 engine using FT weights.
# Enable the special TensorRT-LLM GPT Attention plugin (--use_gpt_attention_plugin) to increase runtime performance.
# It is recommend to use --remove_input_padding along with --use_gpt_attention_plugin for better performance
python3 build.py --model_dir=./c-model/gpt2/1-gpu --use_gpt_attention_plugin --remove_input_padding
```

## 创建 model repository

在 all_models/inflight_batcher_llm 目录中有五个模型：

- "preprocessing"：该模型用于进行 tokenizing，即将 prompt（字符串）转换为 input_ids（整数列表）。
- "tensorrt_llm"：该模型是 TensorRT-LLM 模型的包装器，用于推理。
- "postprocessing"：该模型用于 de-tokenizing，即将 output_ids（整数列表）转换为输出（字符串）。
- "ensemble"：该模型可以将 preprocessing、tensorrt_llm 和 postprocessing 模型链接在一起使用。
- "tensorrt_llm_bls"：该模型将 preprocessing、tensorrt_llm 和 postprocessing 模型链接在一起使用。BLS 模型具有一个可选参数 accumulate_tokens，可以在 streaming 模式下使用，将累积的所有 token 作为输入调用 preprocessing 模型，而不仅仅是一个 token。这对于某些分词器可能是必要的。

BLS 是 `Business Logic Scripting` 的缩写，解释如下：

Triton 的 ensemble 功能支持许多使用情况，可以将多个模型被组合成一个管道（或更一般地说，一个有向无环图）。然而，还有许多其他不支持的使用情况，因为作为模型管道的一部分，它们需要循环、条件语句（if-then-else）、数据相关的控制流和其他自定义逻辑与模型执行相互交织。我们将这种自定义逻辑和模型执行的组合称为 Business Logic Scripting（BLS）。

详细参考[文档](https://github.com/triton-inference-server/python_backend/blob/main/README.md#business-logic-scripting)

```
# Create the model repository that will be used by the Triton server
cd tensorrtllm_backend
mkdir triton_model_repo

# Copy the example models to the model repository
cp -r all_models/inflight_batcher_llm/* triton_model_repo/

# Copy the TRT engine to triton_model_repo/tensorrt_llm/1/
cp tensorrt_llm/examples/gpt/engines/fp16/1-gpu/* triton_model_repo/tensorrt_llm/1
```

## 修改配置参数

*triton_model_repo/preprocessing/config.pbtxt*

|       Name       |                               Description                                |
| :--------------: | :----------------------------------------------------------------------: |
| `tokenizer_dir`  |                         模型的 tokenizer 路径。                          |
| `tokenizer_type` | 模型的 tokenizer 类型, 支持 `t5`, `auto` 和 `llama`。这里可以使用 `auto` |

*triton_model_repo/tensorrt_llm/config.pbtxt*

|               Name               |                                                                                                                                                                                                                                     Description                                                                                                                                                                                                                                     |
| :------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|         `gpt_model_type`         |                                                                                                                                                                                                必填项。 启用 in-flight batching 时设置为 `inflight_fused_batching`， 否则设置为 `V1`                                                                                                                                                                                                |
|         `gpt_model_path`         |                                                                                                                                                                                                                        必填项。TensorRT-LLM engines 的路径。                                                                                                                                                                                                                        |
|     `batch_scheduler_policy`     |                                                                                                                    必填项。设置为 `max_utilization` 来尽可能多的将请求打包为 in-flight batching。这将最大化吞吐量，但是可能会导致执行过程中达到 KV cache 限制而产生开销，因此需要请求暂停/恢复。设置为 `guaranteed_no_evict` 以确保已启动的请求永远不会被暂停。                                                                                                                     |
|           `decoupled`            |                                                                                                                                                                                       可选项（默认为 false）。控制 streaming. 当客户端使用 streaming 选项时，Decoupled 模式必须设置为 `True`                                                                                                                                                                                        |
|         `max_beam_width`         |                                                                                                                                                                                                          可选项（默认为 1 (default=1)。当使用 beam search 时的最大的宽度。                                                                                                                                                                                                          |
|  `max_tokens_in_paged_kv_cache`  |                                                                                                                                     可选项(默认为 unspecified). KV cache 的最大 token 数量. 当为 unspecified 时, 理解为 'infinite'. KV cache 分配是 max_tokens_in_paged_kv_cache 和下面从 kv_cache_free_gpu_mem_fraction 派生的值之间的最小值。                                                                                                                                     |
|   `max_attention_window_size`    |                                                                                                                                                                            可选项(默认为 max_sequence_length)。当使用 sliding window attention 时, attention 窗口的最大值。 默认为当前序列的所有 token。                                                                                                                                                                            |
| `kv_cache_free_gpu_mem_fraction` |                                                                                                                                                                                  可选项 (默认为 0.9)。设置为介于 0 和 1 之间的数值，表示在加载模型后可以用于 KV cache 的 GPU 内存的最大使用比例。                                                                                                                                                                                   |
|       `enable_trt_overlap`       |                                                                                                                                                                                   可选项(默认为 `false`). 将其设置为 true，将可用的请求分割为两个 microbatches，可以并行运行以隐藏 CPU 运行时间。                                                                                                                                                                                   |
|    `exclude_input_in_output`     |                                                                                                                                                                   可选项(默认为 `false`). 将其设置为 true，仅在 response 中返回生成的 token。将其设置为 false，返回由生成的 token 与提示词 token 连接而成的结果。                                                                                                                                                                   |
|      `normalize_log_probs`       |                                                                                                                                                                                                      可选项(默认为 `true`). 设置为 `false` 来跳过`output_log_probs` 的正则化。                                                                                                                                                                                                      |
|     `enable_chunked_context`     |                                                                                                                                                                                                            可选项(默认为 `false`). 将其设置为 `true` 以启用上下文分块。                                                                                                                                                                                                             |
|         `decoding_mode`          | 可选项. 设置为以下之一：{top_k, top_p, top_k_top_p, beam_search}，以选择解码模式。top_k 模式仅使用 Top-K 算法进行采样，top_p 模式仅使用 Top-P 算法进行采样。top_k_top_p 模式根据请求的运行时采样参数同时使用 Top-K 和 Top-P 算法。请注意，top_k_top_p 选项需要更多内存，并且运行时间比单独使用 top_k 或 top_p 更长；因此，只有在必要时才应使用它。beam_search 使用 beam search 算法。如果未指定，默认情况下，如果 max_beam_width == 1，则使用 top_k_top_p；否则，使用 beam_search。 |

*triton_model_repo/postprocessing/config.pbtxt*

|       Name       |                               Description                                |
| :--------------: | :----------------------------------------------------------------------: |
| `tokenizer_dir`  |                         模型的 tokenizer 路径。                          |
| `tokenizer_type` | 模型的 tokenizer 类型, 支持 `t5`, `auto` 和 `llama`。这里可以使用 `auto` |