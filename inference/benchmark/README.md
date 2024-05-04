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
| Llama-2-7b-chat-hf       |
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

当 request-rate=8 时，出现了 Pending request。

```
INFO 05-04 15:15:56 metrics.py:229] Avg prompt throughput: 635.9 tokens/s, Avg generation throughput: 1234.8 tokens/s, Running: 92 reqs, Swapped: 0 reqs, Pending: 323 reqs, GPU KV cache usage: 98.9%, CPU KV cache usage: 0.0%
INFO 05-04 15:16:01 metrics.py:229] Avg prompt throughput: 1518.9 tokens/s, Avg generation throughput: 869.6 tokens/s, Running: 101 reqs, Swapped: 0 reqs, Pending: 292 reqs, GPU KV cache usage: 99.0%, CPU KV cache usage: 0.0%
INFO 05-04 15:16:06 metrics.py:229] Avg prompt throughput: 1467.3 tokens/s, Avg generation throughput: 1036.5 tokens/s, Running: 114 reqs, Swapped: 0 reqs, Pending: 249 reqs, GPU KV cache usage: 98.2%, CPU KV cache usage: 0.0%
INFO 05-04 15:16:12 metrics.py:229] Avg prompt throughput: 1678.5 tokens/s, Avg generation throughput: 1021.6 tokens/s, Running: 113 reqs, Swapped: 0 reqs, Pending: 221 reqs, GPU KV cache usage: 98.0%, CPU KV cache usage: 0.0%
INFO 05-04 15:16:17 metrics.py:229] Avg prompt throughput: 1687.2 tokens/s, Avg generation throughput: 948.2 tokens/s, Running: 119 reqs, Swapped: 0 reqs, Pending: 183 reqs, GPU KV cache usage: 98.9%, CPU KV cache usage: 0.0%
INFO 05-04 15:16:22 metrics.py:229] Avg prompt throughput: 1679.1 tokens/s, Avg generation throughput: 1004.6 tokens/s, Running: 108 reqs, Swapped: 0 reqs, Pending: 162 reqs, GPU KV cache usage: 99.4%, CPU KV cache usage: 0.0%

```

### 参考文献
- [benchmark](https://github.com/fw-ai/benchmark/tree/main)
- [llm-inference-benchmark](https://github.com/ninehills/llm-inference-benchmark/tree/main)
- [vllm benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks)