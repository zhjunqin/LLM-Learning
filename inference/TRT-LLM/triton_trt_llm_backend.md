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

#### preprocessing 的 config.pbtxt

*triton_model_repo/preprocessing/config.pbtxt*

|       Name       |                               Description                                |
| :--------------: | :----------------------------------------------------------------------: |
| `tokenizer_dir`  |                         模型的 tokenizer 路径。                          |
| `tokenizer_type` | 模型的 tokenizer 类型, 支持 `t5`, `auto` 和 `llama`。这里可以使用 `auto` |

#### tensorrt_llm 的 config.pbtxt

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

#### postprocessing 的 config.pbtxt

*triton_model_repo/postprocessing/config.pbtxt*

|       Name       |                               Description                                |
| :--------------: | :----------------------------------------------------------------------: |
| `tokenizer_dir`  |                         模型的 tokenizer 路径。                          |
| `tokenizer_type` | 模型的 tokenizer 类型, 支持 `t5`, `auto` 和 `llama`。这里可以使用 `auto` |

## 启动 server

```
cd /tensorrtllm_backend
# --world_size is the number of GPUs you want to use for serving
python3 scripts/launch_triton_server.py --world_size=1 --model_repo=./triton_model_repo
```

```
I0310 12:50:03.149548 29419 cuda_memory_manager.cc:107] CUDA memory pool is created on device 0 with size 67108864
I0310 12:50:03.158836 29419 model_lifecycle.cc:461] loading: postprocessing:1
I0310 12:50:03.158896 29419 model_lifecycle.cc:461] loading: preprocessing:1
I0310 12:50:03.158943 29419 model_lifecycle.cc:461] loading: tensorrt_llm:1
I0310 12:50:03.158976 29419 model_lifecycle.cc:461] loading: tensorrt_llm_bls:1
I0310 12:50:03.194059 29419 python_be.cc:2362] TRITONBACKEND_ModelInstanceInitialize: postprocessing_0_0 (CPU device 0)
I0310 12:50:03.194071 29419 python_be.cc:2362] TRITONBACKEND_ModelInstanceInitialize: preprocessing_0_0 (CPU device 0)
I0310 12:50:03.258508 29419 python_be.cc:2362] TRITONBACKEND_ModelInstanceInitialize: tensorrt_llm_bls_0_0 (CPU device 0)
[TensorRT-LLM][WARNING] gpu_device_ids is not specified, will be automatically set
[TensorRT-LLM][WARNING] max_beam_width is not specified, will use default value of 1
[TensorRT-LLM][WARNING] max_tokens_in_paged_kv_cache is not specified, will use default value
[TensorRT-LLM][WARNING] batch_scheduler_policy parameter was not found or is invalid (must be max_utilization or guaranteed_no_evict)
[TensorRT-LLM][WARNING] enable_chunked_context is not specified, will be set to false.
[TensorRT-LLM][WARNING] kv_cache_free_gpu_mem_fraction is not specified, will use default value of 0.9 or max_tokens_in_paged_kv_cache
[TensorRT-LLM][WARNING] enable_trt_overlap is not specified, will be set to false
[TensorRT-LLM][WARNING] normalize_log_probs is not specified, will be set to true
[TensorRT-LLM][WARNING] exclude_input_in_output is not specified, will be set to false
[TensorRT-LLM][WARNING] max_attention_window_size is not specified, will use default value (i.e. max_sequence_length)
[TensorRT-LLM][WARNING] enable_kv_cache_reuse is not specified, will be set to false
[TensorRT-LLM][WARNING] Parameter version cannot be read from json:
[TensorRT-LLM][WARNING] [json.exception.out_of_range.403] key 'version' not found
[TensorRT-LLM][INFO] No engine version found in the config file, assuming engine(s) built by old builder API.
[TensorRT-LLM][WARNING] Parameter pipeline_parallel cannot be read from json:
[TensorRT-LLM][WARNING] [json.exception.out_of_range.403] key 'pipeline_parallel' not found
[TensorRT-LLM][WARNING] Parameter head_size cannot be read from json:
[TensorRT-LLM][WARNING] [json.exception.out_of_range.403] key 'head_size' not found
[TensorRT-LLM][WARNING] [json.exception.type_error.302] type must be array, but is null
[TensorRT-LLM][WARNING] Optional value for parameter lora_target_modules will not be set.
[TensorRT-LLM][WARNING] [json.exception.out_of_range.403] key 'mlp_hidden_size' not found
[TensorRT-LLM][WARNING] Optional value for parameter mlp_hidden_size will not be set.
[TensorRT-LLM][INFO] Initializing MPI with thread mode 1
I0310 12:50:03.541997 29419 model_lifecycle.cc:827] successfully loaded 'tensorrt_llm_bls'
[TensorRT-LLM][INFO] MPI size: 1, rank: 0
[TensorRT-LLM][INFO] Rank 0 is using GPU 0
[TensorRT-LLM][INFO] TRTGptModel maxNumSequences: 8
[TensorRT-LLM][INFO] TRTGptModel maxBatchSize: 8
[TensorRT-LLM][INFO] TRTGptModel mMaxAttentionWindowSize: 1024
[TensorRT-LLM][INFO] TRTGptModel enableTrtOverlap: 0
[TensorRT-LLM][INFO] TRTGptModel normalizeLogProbs: 1
[TensorRT-LLM][INFO] Loaded engine size: 777 MiB
[TensorRT-LLM][INFO] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 818, GPU 1129 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] Init cuDNN: CPU +2, GPU +10, now: CPU 820, GPU 1139 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +774, now: CPU 0, GPU 774 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 848, GPU 1329 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] Init cuDNN: CPU +1, GPU +8, now: CPU 849, GPU 1337 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 774 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 867, GPU 1353 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] Init cuDNN: CPU +0, GPU +10, now: CPU 867, GPU 1363 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 774 (MiB)
[TensorRT-LLM][INFO] Allocate 21604859904 bytes for k/v cache.
[TensorRT-LLM][INFO] Using 219776 total tokens in paged KV cache, and 8 blocks per sequence
I0310 12:50:04.774805 29419 model_lifecycle.cc:827] successfully loaded 'tensorrt_llm'
I0310 12:50:05.253728 29419 model_lifecycle.cc:827] successfully loaded 'preprocessing'
I0310 12:50:05.345225 29419 model_lifecycle.cc:827] successfully loaded 'postprocessing'
I0310 12:50:05.345803 29419 model_lifecycle.cc:461] loading: ensemble:1
I0310 12:50:05.346167 29419 model_lifecycle.cc:827] successfully loaded 'ensemble'
I0310 12:50:05.346249 29419 server.cc:606]
+------------------+------+
| Repository Agent | Path |
+------------------+------+
+------------------+------+

I0310 12:50:05.346297 29419 server.cc:633]
+-------------+-----------------------------------------------------------------+---------------------------------------------------------------------------------+
| Backend     | Path                                                            | Config                                                                          |
+-------------+-----------------------------------------------------------------+---------------------------------------------------------------------------------+
| python      | /opt/tritonserver/backends/python/libtriton_python.so           | {"cmdline":{"auto-complete-config":"false","backend-directory":"/opt/tritonserv |
|             |                                                                 | er/backends","min-compute-capability":"6.000000","shm-region-prefix-name":"pref |
|             |                                                                 | ix0_","default-max-batch-size":"4"}}                                            |
| tensorrtllm | /opt/tritonserver/backends/tensorrtllm/libtriton_tensorrtllm.so | {"cmdline":{"auto-complete-config":"false","backend-directory":"/opt/tritonserv |
|             |                                                                 | er/backends","min-compute-capability":"6.000000","default-max-batch-size":"4"}} |
+-------------+-----------------------------------------------------------------+---------------------------------------------------------------------------------+

I0310 12:50:05.346332 29419 server.cc:676]
+------------------+---------+--------+
| Model            | Version | Status |
+------------------+---------+--------+
| ensemble         | 1       | READY  |
| postprocessing   | 1       | READY  |
| preprocessing    | 1       | READY  |
| tensorrt_llm     | 1       | READY  |
| tensorrt_llm_bls | 1       | READY  |
+------------------+---------+--------+

I0310 12:50:05.457967 29419 metrics.cc:877] Collecting metrics for GPU 0: NVIDIA GeForce RTX 3090
I0310 12:50:05.474071 29419 metrics.cc:770] Collecting CPU metrics
I0310 12:50:05.474295 29419 tritonserver.cc:2498]
+----------------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| Option                           | Value                                                                                                                          |
+----------------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| server_id                        | triton                                                                                                                         |
| server_version                   | 2.42.0                                                                                                                         |
| server_extensions                | classification sequence model_repository model_repository(unload_dependents) schedule_policy model_configuration system_shared |
|                                  | _memory cuda_shared_memory binary_tensor_data parameters statistics trace logging                                              |
| model_repository_path[0]         | /path/to/triton_model_repo/                                                                                                    |
| model_control_mode               | MODE_NONE                                                                                                                      |
| strict_model_config              | 1                                                                                                                              |
| rate_limit                       | OFF                                                                                                                            |
| pinned_memory_pool_byte_size     | 268435456                                                                                                                      |
| cuda_memory_pool_byte_size{0}    | 67108864                                                                                                                       |
| min_supported_compute_capability | 6.0                                                                                                                            |
| strict_readiness                 | 1                                                                                                                              |
| exit_timeout                     | 30                                                                                                                             |
| cache_enabled                    | 0                                                                                                                              |
+----------------------------------+--------------------------------------------------------------------------------------------------------------------------------+

I0310 12:50:05.517577 29419 grpc_server.cc:2519] Started GRPCInferenceService at 0.0.0.0:8001
I0310 12:50:05.517851 29419 http_server.cc:4623] Started HTTPService at 0.0.0.0:8000
I0310 12:50:05.558898 29419 http_server.cc:315] Started Metrics Service at 0.0.0.0:8002

```
查看进程：
```
# ps -eo pid,etime,cmd
PID     PPID     ELAPSED CMD
29769       1    00:07 mpirun --allow-run-as-root -n 1 /opt/tritonserver/bin/tritonserver --model-repository=./triton_model_repo/ --grpc-port=8001 --http-port=8000
29773   29769    00:07 /opt/tritonserver/bin/tritonserver --model-repository=./triton_model_repo/ --grpc-port=8001 --http-port=8000 --metrics-port=8002 --disable-a
29807   29773    00:07 /opt/tritonserver/backends/python/triton_python_backend_stub ./triton_model_repo/postprocessing/1/model.py prefix0_2 1048576 1048576 29773 /
29808   29773    00:07 /opt/tritonserver/backends/python/triton_python_backend_stub ./triton_model_repo/preprocessing/1/model.py prefix0_1 1048576 1048576 29773 /o
29906   29773    00:07 /opt/tritonserver/backends/python/triton_python_backend_stub ./triton_model_repo/tensorrt_llm_bls/1/model.py prefix0_3 1048576 1048576 29773
```

`scripts/launch_triton_server.py` 生成命令的代码如下：

```
def get_cmd(world_size, tritonserver, grpc_port, http_port, metrics_port,
            model_repo, log, log_file, tensorrt_llm_model_name):
    cmd = ['mpirun', '--allow-run-as-root']
    for i in range(world_size):
        cmd += ['-n', '1', tritonserver, f'--model-repository={model_repo}']
        if log and (i == 0):
            cmd += ['--log-verbose=3', f'--log-file={log_file}']
        # If rank is not 0, skip loading of models other than `tensorrt_llm_model_name`
        if (i != 0):
            cmd += [
                '--model-control-mode=explicit',
                f'--load-model={tensorrt_llm_model_name}'
            ]
        cmd += [
            f'--grpc-port={grpc_port}', f'--http-port={http_port}',
            f'--metrics-port={metrics_port}', '--disable-auto-complete-config',
            f'--backend-config=python,shm-region-prefix-name=prefix{i}_', ':'
        ]
    return cmd
```

## 请求 Server

```
curl -X POST localhost:8000/v2/models/${MODEL_NAME}/generate -d '{"{PARAM1_KEY}": "{PARAM1_VALUE}", ... }'
```

请求 ensemble 模型

```
# curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": ""}' | jq
{
  "context_logits": 0,
  "cum_log_probs": 0,
  "generation_logits": 0,
  "model_name": "ensemble",
  "model_version": "1",
  "output_log_probs": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
  "sequence_end": false,
  "sequence_id": 0,
  "sequence_start": false,
  "text_output": "What is machine learning?\n\nMachine learning is a method of learning by using machine learning algorithms to solve problems.\n\n"
}

```

如果设置了 `exclude_input_in_output` 为 true，则输出中不包括输入的提示词。
```
"text_output": "\n\nMachine learning is a method of learning by using machine learning algorithms to solve problems.\n\n"
```

请求 tensorrt_llm_bls 模型

```
# curl -X POST localhost:8000/v2/models/tensorrt_llm_bls/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": ""}' | jq
{
  "context_logits": 0,
  "cum_log_probs": 0,
  "generation_logits": 0,
  "model_name": "tensorrt_llm_bls",
  "model_version": "1",
  "output_log_probs": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
  "text_output": "\n\nMachine learning is a method of learning by using machine learning algorithms to solve problems.\n\n"
}

```

使用 python 客户端：

```
#  python3 inflight_batcher_llm/client/inflight_batcher_llm_client.py --request-output-len 200 --tokenizer-dir /path/to/gpt2-medium/
=========
Input sequence:  [28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257]
Got completed request
Input: Born in north-east France, Soyer trained as a
Output beam 0:  has since worked in restaurants in London, Paris, Milan and Rome.

He is married to the former model and actress, Anna-Marie, and has two children, a daughter, Emma, and a son, Daniel.

Soyer's wife, Anna-Marie, is a former model and actress.

He is survived by his wife, Anna-Marie, and their two children, Daniel and Emma.

Soyer was born in Paris, France, to a French father and a German mother.

He was educated at the prestigious Ecole des Beaux-Arts in Paris and the Sorbonne in Paris.

He was a member of the French Academy of Sciences and the French Academy of Arts.

He was a member of the French Academy of Sciences and the French Academy of Arts.

Soyer was a member of the French Academy of Sciences and the
Output sequence:  [21221, 878, 3867, 284, 3576, 287, 262, 1903, 6303, 82, 13, 679, 468, 1201, 3111, 287, 10808, 287, 3576, 11, 6342, 11, 21574, 290, 10598, 13, 198, 198, 1544, 318, 6405, 284, 262, 1966, 2746, 290, 14549, 11, 11735, 12, 44507, 11, 290, 468, 734, 1751, 11, 257, 4957, 11, 18966, 11, 290, 257, 3367, 11, 7806, 13, 198, 198, 50, 726, 263, 338, 3656, 11, 11735, 12, 44507, 11, 318, 257, 1966, 2746, 290, 14549, 13, 198, 198, 1544, 318, 11803, 416, 465, 3656, 11, 11735, 12, 44507, 11, 290, 511, 734, 1751, 11, 7806, 290, 18966, 13, 198, 198, 50, 726, 263, 373, 4642, 287, 6342, 11, 4881, 11, 284, 257, 4141, 2988, 290, 257, 2679, 2802, 13, 198, 198, 1544, 373, 15657, 379, 262, 23566, 38719, 293, 748, 1355, 14644, 12, 3163, 912, 287, 6342, 290, 262, 15423, 4189, 710, 287, 6342, 13, 198, 198, 1544, 373, 257, 2888, 286, 262, 4141, 8581, 286, 13473, 290, 262, 4141, 8581, 286, 11536, 13, 198, 198, 1544, 373, 257, 2888, 286, 262, 4141, 8581, 286, 13473, 290, 262, 4141, 8581, 286, 11536, 13, 198, 198, 50, 726, 263, 373, 257, 2888, 286, 262, 4141, 8581, 286, 13473, 290, 262]
```

## Triton Metrics

```
curl localhost:8002/metrics
```

```
# HELP nv_trt_llm_request_metrics TRT LLM request metrics
# TYPE nv_trt_llm_request_metrics gauge
nv_trt_llm_request_metrics{model="tensorrt_llm",request_type="context",version="1"} 1
nv_trt_llm_request_metrics{model="tensorrt_llm",request_type="scheduled",version="1"} 1
nv_trt_llm_request_metrics{model="tensorrt_llm",request_type="max",version="1"} 512
nv_trt_llm_request_metrics{model="tensorrt_llm",request_type="active",version="1"} 0
# HELP nv_trt_llm_runtime_memory_metrics TRT LLM runtime memory metrics
# TYPE nv_trt_llm_runtime_memory_metrics gauge
nv_trt_llm_runtime_memory_metrics{memory_type="pinned",model="tensorrt_llm",version="1"} 0
nv_trt_llm_runtime_memory_metrics{memory_type="gpu",model="tensorrt_llm",version="1"} 1610236
nv_trt_llm_runtime_memory_metrics{memory_type="cpu",model="tensorrt_llm",version="1"} 0
# HELP nv_trt_llm_kv_cache_block_metrics TRT LLM KV cache block metrics
# TYPE nv_trt_llm_kv_cache_block_metrics gauge
nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="tokens_per",model="tensorrt_llm",version="1"} 64
nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="used",model="tensorrt_llm",version="1"} 1
nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="free",model="tensorrt_llm",version="1"} 6239
nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="max",model="tensorrt_llm",version="1"} 6239
# HELP nv_trt_llm_inflight_batcher_metrics TRT LLM inflight_batcher-specific metrics
# TYPE nv_trt_llm_inflight_batcher_metrics gauge
nv_trt_llm_inflight_batcher_metrics{inflight_batcher_specific_metric="micro_batch_id",model="tensorrt_llm",version="1"} 0
nv_trt_llm_inflight_batcher_metrics{inflight_batcher_specific_metric="generation_requests",model="tensorrt_llm",version="1"} 0
nv_trt_llm_inflight_batcher_metrics{inflight_batcher_specific_metric="total_context_tokens",model="tensorrt_llm",version="1"} 0
# HELP nv_trt_llm_general_metrics General TRT LLM metrics
# TYPE nv_trt_llm_general_metrics gauge
nv_trt_llm_general_metrics{general_type="iteration_counter",model="tensorrt_llm",version="1"} 0
nv_trt_llm_general_metrics{general_type="timestamp",model="tensorrt_llm",version="1"} 1700074049
```

## Benchmark

Tools 目录下有相关工具

```
# ll tools/gpt/
total 56
drwxr-xr-x 2 root root  4096 Mar 10 03:38 ./
drwxr-xr-x 5 root root  4096 Mar  7 01:38 ../
-rw-r--r-- 1 root root  6733 Feb  1 11:20 benchmark_core_model.py
-rw-r--r-- 1 root root  5291 Feb  1 11:20 client.py
-rw-r--r-- 1 root root  6075 Feb  1 11:20 client_async.py
-rw-r--r-- 1 root root 12433 Feb  1 11:20 end_to_end_test.py
-rw-r--r-- 1 root root  4346 Feb  1 11:20 gen_input_data.py
```

```
# python benchmark_core_model.py -i grpc -b 1 --output_len 200 --topp 0.6 --start_len 24 --num_runs 3
[INFO] Batch size: 1, Start len: 24, Output len: 200
[INFO] Latency: 663.176 ms
[INFO] Throughput: 1.508 sentences / sec
```