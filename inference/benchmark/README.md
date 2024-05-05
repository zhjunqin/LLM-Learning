# Benchmark

对不同的推理框架 benchmark，基于 vllm benchmarks。

## 测试框架/硬件/模型

### 框架列表

| 框架     | 版本        | 地址                                   |
| -------- | ----------- | -------------------------------------- |
| vllm     | 0.4.1       | https://github.com/vllm-project/vllm   |
| lmdeploy | 0.4.0       | https://github.com/InternLM/lmdeploy   |
| TRT-LLM  | 0.9.0       | https://github.com/NVIDIA/TensorRT-LLM |
| mlc-llm  | 0.1.dev1166 | https://github.com/mlc-ai/mlc-llm      |


### 硬件

| GPU  | 显存 | 驱动 | CUDA |
| ---- | ---- | ---- | ---- |
| 3090 | 24G  | 535  | 12.2 |

### 模型

| 模型                     |
| ------------------------ |
| Meta-Llama-3-8B-Instruct |

## 测试流程

### vLLM

启动 vllm

```
python -m vllm.entrypoints.openai.api_server --model Meta-Llama-3-8B-Instruct --dtype auto  --disable-log-requests

INFO 05-04 14:43:16 api_server.py:151] vLLM API server version 0.4.1
INFO 05-04 14:43:16 api_server.py:152] args: Namespace(host=None, port=8000, uvicorn_log_level='info', allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, served_model_name=None, lora_modules=None, chat_template=None, response_role='assistant', ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_cert_reqs=0, root_path=None, middleware=[], model='Meta-Llama-3-8B-Instruct', tokenizer=None, skip_tokenizer_init=False, revision=None, code_revision=None, tokenizer_revision=None, tokenizer_mode='auto', trust_remote_code=False, download_dir=None, load_format='auto', dtype='auto', kv_cache_dtype='auto', quantization_param_path=None, max_model_len=None, guided_decoding_backend='outlines', worker_use_ray=False, pipeline_parallel_size=1, tensor_parallel_size=1, max_parallel_loading_workers=None, ray_workers_use_nsight=False, block_size=16, enable_prefix_caching=False, use_v2_block_manager=False, num_lookahead_slots=0, seed=0, swap_space=4, gpu_memory_utilization=0.9, num_gpu_blocks_override=None, max_num_batched_tokens=None, max_num_seqs=256, max_logprobs=5, disable_log_stats=False, quantization=None, enforce_eager=False, max_context_len_to_capture=8192, disable_custom_all_reduce=False, tokenizer_pool_size=0, tokenizer_pool_type='ray', tokenizer_pool_extra_config=None, enable_lora=False, max_loras=1, max_lora_rank=16, lora_extra_vocab_size=256, lora_dtype='auto', max_cpu_loras=None, device='auto', image_input_type=None, image_token_id=None, image_input_shape=None, image_feature_size=None, scheduler_delay_factor=0.0, enable_chunked_prefill=False, speculative_model=None, num_speculative_tokens=None, speculative_max_model_len=None, model_loader_extra_config=None, engine_use_ray=False, disable_log_requests=True, max_log_len=None)
INFO 05-04 14:43:16 llm_engine.py:98] Initializing an LLM engine (v0.4.1) with config: model='Meta-Llama-3-8B-Instruct', speculative_config=None, tokenizer='Meta-Llama-3-8B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=auto, tensor_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), seed=0)
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
INFO 05-04 14:43:16 utils.py:608] Found nccl from library /root/.config/vllm/nccl/cu12/libnccl.so.2.18.1
INFO 05-04 14:43:17 selector.py:77] Cannot use FlashAttention backend because the flash_attn package is not found. Please install it for better performance.
INFO 05-04 14:43:17 selector.py:33] Using XFormers backend.
[rank0]:[W ProcessGroupGloo.cpp:721] Warning: Unable to resolve hostname to a (local) address. Using the loopback address as fallback. Manually set the network interface to bind to with GLOO_SOCKET_IFNAME. (function operator())
INFO 05-04 14:43:20 model_runner.py:173] Loading model weights took 14.9595 GB
INFO 05-04 14:43:23 gpu_executor.py:119] # GPU blocks: 2358, # CPU blocks: 2048
INFO 05-04 14:43:24 model_runner.py:976] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 05-04 14:43:24 model_runner.py:980] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 05-04 14:43:29 model_runner.py:1057] Graph capturing finished in 5 secs.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
INFO 05-04 14:43:29 serving_chat.py:344] Using default chat template:
INFO 05-04 14:43:29 serving_chat.py:344] {% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>
INFO 05-04 14:43:29 serving_chat.py:344]
INFO 05-04 14:43:29 serving_chat.py:344] '+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>
INFO 05-04 14:43:29 serving_chat.py:344]
INFO 05-04 14:43:29 serving_chat.py:344] ' }}{% else %}{{ eos_token }}{% endif %}
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
INFO:     Started server process [23347]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

```


benchmark 命令

```
python benchmarks/benchmark_serving.py         --backend vllm         --model "Meta-Llama-3-8B-Instruct"         --dataset-name sharegpt         --dataset-path "ShareGPT_V3_unfiltered_cleaned_split.json"         --request-rate 1         --num-prompts 100 --result-dir /data/benchmark_result/vllm/  --save-result --metadata backend=vllm request-rate=1 num-prompts=100
 ```

#### Benchmark 结果

| request-rate | num-prompts | duration (s) | Total input tokens | Total generated tokens | Request throughput (req/s) | Input token throughput (tok/s) | Output token throughput (tok/s) | P50 TTFT (ms) | P90 TTFT (ms) | P99 TTFT (ms) | P50 TPOT (ms) | P90 TPOT (ms) | P99 TPOT (ms) |
| ------------ | ----------- | ------------ | ------------------ | ---------------------- | -------------------------- | ------------------------------ | ------------------------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 1            | 100         | 105.08       | 22925              | 21789                  | 0.95                       | 218.17                         | 207.35                          | 68.75         | 172.66        | 216.07        | 24.68         | 26.86         | 33.59         |
| 2            | 200         | 112.79       | 42889              | 44132                  | 1.77                       | 380.26                         | 391.28                          | 68.18         | 186.67        | 244.08        | 29.64         | 33.58         | 45.39         |
| 5            | 1000        | 218.21       | 213987             | 199774                 | 4.58                       | 980.65                         | 915.52                          | 119.85        | 232.46        | 370.85        | 82.95         | 82.95         | 96.64         |
| 8            | 1000        | 202.53       | 213987             | 199774                 | 4.94                       | 1056.55                        | 986.37                          | 24919.84      | 49774.61      | 53385.48      | 104.71        | 128.04        | 199.54        |

当 request-rate=8 时，出现了 Pending request。从上面也可以看到排队导致 TTFT 变长非常多。

```
INFO 05-04 15:15:56 metrics.py:229] Avg prompt throughput: 635.9 tokens/s, Avg generation throughput: 1234.8 tokens/s, Running: 92 reqs, Swapped: 0 reqs, Pending: 323 reqs, GPU KV cache usage: 98.9%, CPU KV cache usage: 0.0%
INFO 05-04 15:16:01 metrics.py:229] Avg prompt throughput: 1518.9 tokens/s, Avg generation throughput: 869.6 tokens/s, Running: 101 reqs, Swapped: 0 reqs, Pending: 292 reqs, GPU KV cache usage: 99.0%, CPU KV cache usage: 0.0%
INFO 05-04 15:16:06 metrics.py:229] Avg prompt throughput: 1467.3 tokens/s, Avg generation throughput: 1036.5 tokens/s, Running: 114 reqs, Swapped: 0 reqs, Pending: 249 reqs, GPU KV cache usage: 98.2%, CPU KV cache usage: 0.0%
INFO 05-04 15:16:12 metrics.py:229] Avg prompt throughput: 1678.5 tokens/s, Avg generation throughput: 1021.6 tokens/s, Running: 113 reqs, Swapped: 0 reqs, Pending: 221 reqs, GPU KV cache usage: 98.0%, CPU KV cache usage: 0.0%
INFO 05-04 15:16:17 metrics.py:229] Avg prompt throughput: 1687.2 tokens/s, Avg generation throughput: 948.2 tokens/s, Running: 119 reqs, Swapped: 0 reqs, Pending: 183 reqs, GPU KV cache usage: 98.9%, CPU KV cache usage: 0.0%
INFO 05-04 15:16:22 metrics.py:229] Avg prompt throughput: 1679.1 tokens/s, Avg generation throughput: 1004.6 tokens/s, Running: 108 reqs, Swapped: 0 reqs, Pending: 162 reqs, GPU KV cache usage: 99.4%, CPU KV cache usage: 0.0%
```

上面未使用 flash attention，使用了 flash attention 后并没有提升，反而有点下降。


| request-rate | num-prompts | duration (s) | Total input tokens | Total generated tokens | Request throughput (req/s) | Input token throughput (tok/s) | Output token throughput (tok/s) | P50 TTFT (ms) | P90 TTFT (ms) | P99 TTFT (ms) | P50 TPOT (ms) | P90 TPOT (ms) | P99 TPOT (ms) |
| ------------ | ----------- | ------------ | ------------------ | ---------------------- | -------------------------- | ------------------------------ | ------------------------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 5            | 1000        | 218.53       | 213987             | 199778                 | 4.58                       | 979.21                         | 914.19                          | 123.25        | 231.46        | 370.58        | 84.14         | 98.26         | 158.75        |

使用 AWQ 量化后的性能，也下降了。

| request-rate | num-prompts | duration (s) | Total input tokens | Total generated tokens | Request throughput (req/s) | Input token throughput (tok/s) | Output token throughput (tok/s) | P50 TTFT (ms) | P90 TTFT (ms) | P99 TTFT (ms) | P50 TPOT (ms) | P90 TPOT (ms) | P99 TPOT (ms) |
| ------------ | ----------- | ------------ | ------------------ | ---------------------- | -------------------------- | ------------------------------ | ------------------------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 5            | 1000        | 230.33       | 213987             | 199771                 | 4.34                       | 929.05                         | 867.33                          | 180.10        | 321.00        | 538.60        | 139.76        | 219.49        | 298.66        |


### lmdeploy

启动 lmdeploy

```
# lmdeploy serve api_server Meta-Llama-3-8B-Instruct   --server-port 23333
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
[WARNING] gemm_config.in is not found; using default GEMM algo
HINT:    Please open http://0.0.0.0:23333 in a browser for detailed api usage!!!
HINT:    Please open http://0.0.0.0:23333 in a browser for detailed api usage!!!
HINT:    Please open http://0.0.0.0:23333 in a browser for detailed api usage!!!
INFO:     Started server process [24522]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:23333 (Press CTRL+C to quit)

```

测试命令

```
python benchmarks/benchmark_serving.py         --backend lmdeploy         --model "llama3"          --dataset-name sharegpt         --dataset-path "ShareGPT_V3_unfiltered_cleaned_split.json"         --request-rate 2         --num-prompts 200  --port 23333 --endpoint /v1/completions --tokenizer "Meta-Llama-3-8B-Instruct" --result-dir /benchmark_result/lmdeploy  --save-result --metadata backend=lmdeploy request-rate=2 num-prompts=200 
```

#### Benchmark 结果


| request-rate | num-prompts | duration (s) | Total input tokens | Total generated tokens | Request throughput (req/s) | Input token throughput (tok/s) | Output token throughput (tok/s) | P50 TTFT (ms) | P90 TTFT (ms) | P99 TTFT (ms) | P50 TPOT (ms) | P90 TPOT (ms) | P99 TPOT (ms) |
| ------------ | ----------- | ------------ | ------------------ | ---------------------- | -------------------------- | ------------------------------ | ------------------------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 1            | 100         | 104.29       | 22925              | 20520                  | 0.96                       | 219.83                         | 196.77                          | 69.06         | 182.00        | 216.00        | 22.40         | 24.54         | 28.94         |
| 2            | 200         | 111.77       | 42889              | 41697                  | 1.79                       | 383.73                         | 373.07                          | 67.81         | 195.30        | 267.04        | 26.59         | 29.49         | 37.77         |
| 5            | 1000        | 212.51       | 213987             | 187896                 | 4.71                       | 1006.94                        | 884.16                          | 97.06         | 230.58        | 369.00        | 32.18         | 44.24         | 68.68         |
| 8            | 1000        | 144.06       | 213987             | 187694                 | 6.94                       | 1485.41                        | 1302.89                         | 217.03        | 515.88        | 3661.76       | 74.94         | 94.23         | 141.89        |

kv cache 量化

```
lmdeploy serve api_server Meta-Llama-3-8B-Instruct --quant-policy 8
```


| request-rate | num-prompts | duration (s) | Total input tokens | Total generated tokens | Request throughput (req/s) | Input token throughput (tok/s) | Output token throughput (tok/s) | P50 TTFT (ms) | P90 TTFT (ms) | P99 TTFT (ms) | P50 TPOT (ms) | P90 TPOT (ms) | P99 TPOT (ms) |
| ------------ | ----------- | ------------ | ------------------ | ---------------------- | -------------------------- | ------------------------------ | ------------------------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 5            | 1000        | 212.48       | 213987             | 188221                 | 4.71                       | 1007.09                        | 885.82                          | 93.89         | 230.55        | 376.01        | 31.07         | 42.60         | 67.52         |
| 8            | 1000        | 142.92       | 213987             | 186820                 | 7.00                       | 1497.24                        | 1307.16                         | 184.06        | 376.53        | 1024.51       | 68.76         | 86.98         | 136.48        |


### mlc-llm

```
#  mlc_llm serve /data/mlc-llm/dist/llama3 --model-lib-path /data/mlc-llm/dist/libs/Meta-Llama-3-8B-Instruct-MLC-q4f16_1-cuda.so  --max-batch-size 48 --gpu-memory-utilization 0.90 --mode server
[2024-05-05 02:50:13] INFO auto_device.py:79: Found device: cuda:0
[2024-05-05 02:50:14] INFO auto_device.py:88: Not found device: rocm:0
[2024-05-05 02:50:16] INFO auto_device.py:88: Not found device: metal:0
[2024-05-05 02:50:17] INFO auto_device.py:88: Not found device: vulkan:0
[2024-05-05 02:50:19] INFO auto_device.py:88: Not found device: opencl:0
[2024-05-05 02:50:19] INFO auto_device.py:35: Using device: cuda:0
[2024-05-05 02:50:19] INFO chat_module.py:379: Using model folder: /data/mlc-llm/dist/llama3
[2024-05-05 02:50:19] INFO chat_module.py:380: Using mlc chat config: /data/mlc-llm/dist/llama3/mlc-chat-config.json
[2024-05-05 02:50:19] INFO chat_module.py:529: Using library model: /data/mlc-llm/dist/libs/Meta-Llama-3-8B-Instruct-MLC-q4f16_1-cuda.so
[2024-05-05 02:50:22] INFO engine_base.py:489: Under mode "local", max batch size 48 is specified by user, max KV cache token capacity is set to 8192, prefill chunk size is set to 8192. We choose small max batch size and KV cache capacity to use less GPU memory.
[2024-05-05 02:50:24] INFO engine_base.py:489: Under mode "interactive", max batch size 48 is specified by user, max KV cache token capacity is set to 8192, prefill chunk size is set to 8192. We fix max batch size to 1 for interactive single sequence use.
[2024-05-05 02:50:27] INFO engine_base.py:489: Under mode "server", max batch size 48 is specified by user, max KV cache token capacity is set to 126754, prefill chunk size is set to 8192. We use as much GPU memory as possible (within the limit of gpu_memory_utilization).
[2024-05-05 02:50:27] INFO engine_base.py:527: The actual engine mode is "server". So max batch size is 48, max KV cache token capacity is 126754, prefill chunk size is 8192.
[2024-05-05 02:50:27] INFO engine_base.py:536: Estimated total single GPU memory usage: 21833.68 MB (Parameters: 4308.13 MB. KVCache: 16054.47 MB. Temporary buffer: 1471.08 MB). The actual usage might be slightly larger than the estimated number.
[2024-05-05 02:50:27] INFO engine_base.py:551: Please switch to mode "local" or "interactive" if you want to use less GPU memory or do not have many concurrent requests to process. Please override the arguments if you have particular values to set.
INFO:     Started server process [29336]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)

```

测试命令

```
python benchmarks/benchmark_serving.py         --backend mlc-llm         --model "Meta-Llama-3-8B-Instruct"         --dataset-name sharegpt         --dataset-path "/data/vllm/ShareGPT_V3_unfiltered_cleaned_split.json"         --request-rate 2         --num-prompts 200 --endpoint /v1/chat/completions --result-dir /benchmark_result/mlc-llm  --save-result --metadata backend=mlc-llm request-rate=2 num-prompts=200 quant=w4a16
```


| request-rate | num-prompts | duration (s) | Total input tokens | Total generated tokens | Request throughput (req/s) | Input token throughput (tok/s) | Output token throughput (tok/s) | P50 TTFT (ms) | P90 TTFT (ms) | P99 TTFT (ms) | P50 TPOT (ms) | P90 TPOT (ms) | P99 TPOT (ms) |
| ------------ | ----------- | ------------ | ------------------ | ---------------------- | -------------------------- | ------------------------------ | ------------------------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 1            | 100         | 100.11       | 22925              | 20586                  | 1.00                       | 228.99                         | 205.63                          | 63.30         | 149.08        | 171.03        | 12.83         | 29.01         | 34.72         |
| 2            | 200         | 108.81       | 42889              | 39678                  | 1.81                       | 393.48                         | 364.65                          | 71.64         | 171.24        | 226.43        | 36.07         | 40.28         | 53.17         |
| 5            | 1000        | 215.88       | 213987             | 179858                 | 4.61                       | 990.13                         | 833.13                          | 2550.71       | 6156.20       | 7806.20       | 52.74         | 62.02         | 85.82         |

在 request-rate=5 时就出现了 TTFT 的延长，说明 mlc-llm 应该没有很好的 continuous batching 的能力。

当启动时设置的  `--max-batch-size` 过大，比如 64 时，就会出现如下错误：

```
Exception in thread Thread-1 (_background_loop):
Traceback (most recent call last):
  File "/opt/conda/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
    self.run()
  File "/opt/conda/lib/python3.10/threading.py", line 953, in run
    self._target(*self._args, **self._kwargs)
  File "/opt/conda/lib/python3.10/site-packages/mlc_llm/serve/engine_base.py", line 1096, in _background_loop
    self._ffi["run_background_loop"]()
  File "tvm/_ffi/_cython/./packed_func.pxi", line 332, in tvm._ffi._cy3.core.PackedFuncBase.__call__
  File "tvm/_ffi/_cython/./packed_func.pxi", line 263, in tvm._ffi._cy3.core.FuncCall
  File "tvm/_ffi/_cython/./packed_func.pxi", line 252, in tvm._ffi._cy3.core.FuncCall3
  File "tvm/_ffi/_cython/./base.pxi", line 182, in tvm._ffi._cy3.core.CHECK_CALL
  File "/opt/conda/lib/python3.10/site-packages/tvm/_ffi/base.py", line 481, in raise_last_ffi_error
    raise py_err
tvm._ffi.base.TVMError: Traceback (most recent call last):
  3: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::contrib::__mk_TVM0::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, tvm::contrib::__mk_TVM0, tvm::runtime::TVMRetValue)
  2: void tvm::contrib::thrust_sort<float, int>(DLTensor*, DLTensor*, DLTensor*, bool, int, DLTensor*) [clone .isra.0]
  1: thrust::detail::temporary_allocator<unsigned char, thrust::detail::execute_with_allocator<thrust::mr::allocator<max_align_t, tvm::contrib::WorkspaceMemoryResource>, thrust::cuda_cub::execute_on_stream_nosync_base> >::allocate(unsigned long)
  0: _ZN3tvm7runtime6deta
  File "/workspace/tvm/src/runtime/contrib/thrust/thrust.cu", line 66
TVMError: Check failed: (result) is false: Failed to allocate 99121664 bytes with alignment 16 bytes.
```

### 参考文献
- [benchmark](https://github.com/fw-ai/benchmark/tree/main)
- [llm-inference-benchmark](https://github.com/ninehills/llm-inference-benchmark/tree/main)
- [vllm benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks)