# 运行和使用 TRT-LLM

执行 [TRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main) 和 [tensorrtllm_backend](https://github.com/triton-inference-server/tensorrtllm_backend/tree/main)，基于版本 v0.8.0。

安装过程参考[这里](./install.md)。

## Clone 代码

Clone tensorrtllm_backend 和 TensorRT-LLM。

```
git clone -b v0.8.0 https://github.com/triton-inference-server/tensorrtllm_backend.git

git submodule update --init --recursive

git lfs install
git lfs pull
```
TensorRT-LLM 作为子模块在 tensorrtllm_backend 子目录中。

## 下载模型

进入到路径 `tensorrtllm_backend/tensorrt_llm/examples/gpt`

从 https://huggingface.co/openai-community/gpt2-medium 下载 gpt2-medium

由于有多个模型文件，这里只下载 bin 模型文件。

```
HF_ENDPOINT='https://hf-mirror.com' huggingface-cli download --local-dir-use-symlinks False --resume-download openai-community/gpt2-medium --local-dir ./gpt2-medium  --include "*.json"
HF_ENDPOINT='https://hf-mirror.com' huggingface-cli download --local-dir-use-symlinks False --resume-download openai-community/gpt2-medium --local-dir ./gpt2-medium  --include "*.bin"
```

## 模型转换

`hf_gpt_convert.py` 脚本将权重从 HF Transformers 格式转换为 FT 格式。

```
python3 hf_gpt_convert.py -i ./gpt2-medium/ -o ./c-model/gpt2 --tensor-parallelism 1 --storage-type float16
```

生成的权重文件：
```
-rw-r--r-- 1 root root      2331 Mar 10 02:26 config.ini
-rw-r--r-- 1 root root      2048 Mar 10 02:26 model.final_layernorm.bias.bin
-rw-r--r-- 1 root root      2048 Mar 10 02:26 model.final_layernorm.weight.bin
-rw-r--r-- 1 root root      2048 Mar 10 02:26 model.layers.0.attention.dense.bias.bin
-rw-r--r-- 1 root root   2097152 Mar 10 02:26 model.layers.0.attention.dense.weight.0.bin
-rw-r--r-- 1 root root      6144 Mar 10 02:26 model.layers.0.attention.query_key_value.bias.0.bin
-rw-r--r-- 1 root root   6291456 Mar 10 02:26 model.layers.0.attention.query_key_value.weight.0.bin
-rw-r--r-- 1 root root      2048 Mar 10 02:26 model.layers.0.input_layernorm.bias.bin
-rw-r--r-- 1 root root      2048 Mar 10 02:26 model.layers.0.input_layernorm.weight.bin
-rw-r--r-- 1 root root      2048 Mar 10 02:26 model.layers.0.mlp.dense_4h_to_h.bias.bin
-rw-r--r-- 1 root root   8388608 Mar 10 02:26 model.layers.0.mlp.dense_4h_to_h.weight.0.bin
-rw-r--r-- 1 root root      8192 Mar 10 02:26 model.layers.0.mlp.dense_h_to_4h.bias.0.bin
-rw-r--r-- 1 root root   8388608 Mar 10 02:26 model.layers.0.mlp.dense_h_to_4h.weight.0.bin
-rw-r--r-- 1 root root      2048 Mar 10 02:26 model.layers.0.post_attention_layernorm.bias.bin
-rw-r--r-- 1 root root      2048 Mar 10 02:26 model.layers.0.post_attention_layernorm.weight.bin
-rw-r--r-- 1 root root      2048 Mar 10 02:26 model.layers.1.attention.dense.bias.bin
-rw-r--r-- 1 root root   2097152 Mar 10 02:26 model.layers.1.attention.dense.weight.0.bin
-rw-r--r-- 1 root root      6144 Mar 10 02:26 model.layers.1.attention.query_key_value.bias.0.bin
-rw-r--r-- 1 root root   6291456 Mar 10 02:26 model.layers.1.attention.query_key_value.weight.0.bin
-rw-r--r-- 1 root root      2048 Mar 10 02:26 model.layers.1.input_layernorm.bias.bin
-rw-r--r-- 1 root root      2048 Mar 10 02:26 model.layers.1.input_layernorm.weight.bin
-rw-r--r-- 1 root root      2048 Mar 10 02:26 model.layers.1.mlp.dense_4h_to_h.bias.bin
-rw-r--r-- 1 root root   8388608 Mar 10 02:26 model.layers.1.mlp.dense_4h_to_h.weight.0.bin
-rw-r--r-- 1 root root      8192 Mar 10 02:26 model.layers.1.mlp.dense_h_to_4h.bias.0.bin
-rw-r--r-- 1 root root   8388608 Mar 10 02:26 model.layers.1.mlp.dense_h_to_4h.weight.0.bin
-rw-r--r-- 1 root root      2048 Mar 10 02:26 model.layers.1.post_attention_layernorm.bias.bin
-rw-r--r-- 1 root root      2048 Mar 10 02:26 model.layers.1.post_attention_layernorm.weight.bin
...省略
```

如果使用 `--tensor-parallelism 2` 生成的权重文件如下，被切分的 Tensor 后面有 idx 作为后缀来标识：

```
python3 hf_gpt_convert.py -i ./gpt2-medium/ -o ./c-model/gpt2 --tensor-parallelism 2 --storage-type float16
```

```
-rw-r--r-- 1 root root      2331 Mar 10 08:18 config.ini
-rw-r--r-- 1 root root      2048 Mar 10 08:18 model.final_layernorm.bias.bin
-rw-r--r-- 1 root root      2048 Mar 10 08:18 model.final_layernorm.weight.bin
-rw-r--r-- 1 root root      2048 Mar 10 08:18 model.layers.0.attention.dense.bias.bin
-rw-r--r-- 1 root root   1048576 Mar 10 08:18 model.layers.0.attention.dense.weight.0.bin
-rw-r--r-- 1 root root   1048576 Mar 10 08:18 model.layers.0.attention.dense.weight.1.bin
-rw-r--r-- 1 root root      3072 Mar 10 08:18 model.layers.0.attention.query_key_value.bias.0.bin
-rw-r--r-- 1 root root      3072 Mar 10 08:18 model.layers.0.attention.query_key_value.bias.1.bin
-rw-r--r-- 1 root root   3145728 Mar 10 08:18 model.layers.0.attention.query_key_value.weight.0.bin
-rw-r--r-- 1 root root   3145728 Mar 10 08:18 model.layers.0.attention.query_key_value.weight.1.bin
-rw-r--r-- 1 root root      2048 Mar 10 08:18 model.layers.0.input_layernorm.bias.bin
-rw-r--r-- 1 root root      2048 Mar 10 08:18 model.layers.0.input_layernorm.weight.bin
-rw-r--r-- 1 root root      2048 Mar 10 08:18 model.layers.0.mlp.dense_4h_to_h.bias.bin
-rw-r--r-- 1 root root   4194304 Mar 10 08:18 model.layers.0.mlp.dense_4h_to_h.weight.0.bin
-rw-r--r-- 1 root root   4194304 Mar 10 08:18 model.layers.0.mlp.dense_4h_to_h.weight.1.bin
-rw-r--r-- 1 root root      4096 Mar 10 08:18 model.layers.0.mlp.dense_h_to_4h.bias.0.bin
-rw-r--r-- 1 root root      4096 Mar 10 08:18 model.layers.0.mlp.dense_h_to_4h.bias.1.bin
-rw-r--r-- 1 root root   4194304 Mar 10 08:18 model.layers.0.mlp.dense_h_to_4h.weight.0.bin
-rw-r--r-- 1 root root   4194304 Mar 10 08:18 model.layers.0.mlp.dense_h_to_4h.weight.1.bin
-rw-r--r-- 1 root root      2048 Mar 10 08:18 model.layers.0.post_attention_layernorm.bias.bin
-rw-r--r-- 1 root root      2048 Mar 10 08:18 model.layers.0.post_attention_layernorm.weight.bin
-rw-r--r-- 1 root root      2048 Mar 10 08:18 model.layers.1.attention.dense.bias.bin
-rw-r--r-- 1 root root   1048576 Mar 10 08:18 model.layers.1.attention.dense.weight.0.bin
-rw-r--r-- 1 root root   1048576 Mar 10 08:18 model.layers.1.attention.dense.weight.1.bin
-rw-r--r-- 1 root root      3072 Mar 10 08:18 model.layers.1.attention.query_key_value.bias.0.bin
-rw-r--r-- 1 root root      3072 Mar 10 08:18 model.layers.1.attention.query_key_value.bias.1.bin
-rw-r--r-- 1 root root   3145728 Mar 10 08:18 model.layers.1.attention.query_key_value.weight.0.bin
-rw-r--r-- 1 root root   3145728 Mar 10 08:18 model.layers.1.attention.query_key_value.weight.1.bin
-rw-r--r-- 1 root root      2048 Mar 10 08:18 model.layers.1.input_layernorm.bias.bin
-rw-r--r-- 1 root root      2048 Mar 10 08:18 model.layers.1.input_layernorm.weight.bin
-rw-r--r-- 1 root root      2048 Mar 10 08:18 model.layers.1.mlp.dense_4h_to_h.bias.bin
-rw-r--r-- 1 root root   4194304 Mar 10 08:18 model.layers.1.mlp.dense_4h_to_h.weight.0.bin
-rw-r--r-- 1 root root   4194304 Mar 10 08:18 model.layers.1.mlp.dense_4h_to_h.weight.1.bin
-rw-r--r-- 1 root root      4096 Mar 10 08:18 model.layers.1.mlp.dense_h_to_4h.bias.0.bin
-rw-r--r-- 1 root root      4096 Mar 10 08:18 model.layers.1.mlp.dense_h_to_4h.bias.1.bin
-rw-r--r-- 1 root root   4194304 Mar 10 08:18 model.layers.1.mlp.dense_h_to_4h.weight.0.bin
-rw-r--r-- 1 root root   4194304 Mar 10 08:18 model.layers.1.mlp.dense_h_to_4h.weight.1.bin
-rw-r--r-- 1 root root      2048 Mar 10 08:18 model.layers.1.post_attention_layernorm.bias.bin
-rw-r--r-- 1 root root      2048 Mar 10 08:18 model.layers.1.post_attention_layernorm.weight.bin
...省略
```

## 编译 TRT engine

```
# Build a single-GPU float16 engine using FT weights.
# Enable the special TensorRT-LLM GPT Attention plugin (--use_gpt_attention_plugin) to increase runtime performance.
# It is recommend to use --remove_input_padding along with --use_gpt_attention_plugin for better performance
python3 build.py --model_dir=./c-model/gpt2/1-gpu --use_gpt_attention_plugin --remove_input_padding
```

生成的文件：
```
-rw-r--r-- 1 root root 1.7K Mar 10 02:34 config.json
-rw-r--r-- 1 root root 778M Mar 10 02:34 gpt_float16_tp1_rank0.engine
```


编译生成 `--tensor-parallelism 2` 的 engine，设置参数 `--world_size=2` ：

```
python3 build.py --model_dir=./c-model/gpt2/2-gpu \
                --remove_input_padding \
                --use_gpt_attention_plugin \
                --enable_context_fmha \
                --use_gemm_plugin \
                --max_batch_size 8 \
                --max_input_len 924 \
                --max_output_len 100 \
                --output_dir trt_engine/gpt2/fp16/2-gpu/ \
                --hidden_act gelu \
                --use_inflight_batching \
                --world_size=2
 ```

生成的 engine 文件：
```
-rw-r--r-- 1 root root 1.7K Mar 10 08:31 config.json
-rw-r--r-- 1 root root 441M Mar 10 08:31 gpt_float16_tp2_rank0.engine
-rw-r--r-- 1 root root 441M Mar 10 08:31 gpt_float16_tp2_rank1.engine
```

对比两个 config.json，使用 Tensor Parallel 的，增加了 nccl plugin。
```
27c27
<     "tensor_parallel": 1,
---
>     "tensor_parallel": 2,
45c45
<     "nccl_plugin": null,
---
>     "nccl_plugin": "float16",
```

#### Fused MultiHead Attention (FMHA)
通过在 build.py 的调用中添加 `--enable_context_fmha`，可以启用 GPT 的 FMHA kernel。

如果发现默认的 fp16 累加（--enable_context_fmha）无法满足需求，你可以尝试启用 fp32 累加，通过添加 `--enable_context_fmha_fp32_acc`。然而，预计会出现性能下降。

请注意，`--enable_context_fmha` 或 `--enable_context_fmha_fp32_acc` 必须与 `--use_gpt_attention_plugin float16` 一起使用。

#### In-flight batching and paged KV cache
如果想在 C++ runtime 中使用 in-flight batching，必须相应地构建 engine。可以通过在 build.py 的调用中添加 `--use_inflight_batching` 来启用 in-flight batching。

请注意，C++ runtime 中的 in-flight batching 仅适用于具有以下特性：attention plugin --use_gpt_attention_plugin=float16、paged KV cache --paged_kv_cache 和 packed data --remove_input_padding。添加 --use_inflight_batching 将自动启用这三个参数。

可以选择不同的精度用于 --use_gpt_attention_plugin 。也可以使用 --tokens_per_block=N 来进一步控制 paged KV cache 中块的大小。

build.py 代码如下：
```
    if args.use_inflight_batching:
        if not args.use_gpt_attention_plugin:
            args.use_gpt_attention_plugin = 'float16'
            logger.info(
                f"Using GPT attention plugin for inflight batching mode. Setting to default '{args.use_gpt_attention_plugin}'"
            )
        if not args.remove_input_padding:
            args.remove_input_padding = True
            logger.info(
                "Using remove input padding for inflight batching mode.")
        if not args.paged_kv_cache:
            args.paged_kv_cache = True
            logger.info("Using paged KV cache for inflight batching mode.")
```

## 运行

#### 单节点，单 GPU

```
# Run the GPT-350M model on a single GPU.
python3 ../run.py --max_output_len=20 --no_add_special_tokens --engine_dir trt_engine/gpt2/fp16/1-gpu/
```

```
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
[TensorRT-LLM][INFO] MPI size: 1, rank: 0
[TensorRT-LLM][INFO] Loaded engine size: 777 MiB
[TensorRT-LLM][INFO] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 920, GPU 1065 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] Init cuDNN: CPU +2, GPU +10, now: CPU 922, GPU 1075 (MiB)
[TensorRT-LLM][WARNING] TensorRT was linked against cuDNN 8.9.6 but loaded cuDNN 8.9.2
[TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +774, now: CPU 0, GPU 774 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 950, GPU 1265 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] Init cuDNN: CPU +0, GPU +8, now: CPU 950, GPU 1273 (MiB)
[TensorRT-LLM][WARNING] TensorRT was linked against cuDNN 8.9.6 but loaded cuDNN 8.9.2
[TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 774 (MiB)
[TensorRT-LLM][INFO] Allocate 21705523200 bytes for k/v cache.
[TensorRT-LLM][INFO] Using 220800 tokens in paged KV cache.
[TensorRT-LLM] TensorRT-LLM version: 0.8.0Input [Text 0]: "Born in north-east France, Soyer trained as a"
Output [Text 0 Beam 0]: " chef before moving to London in the early 1990s. He has since worked in restaurants in London,"
```

#### 单节点，多 GPU

```
mpirun --allow-run-as-root -np 2 python3 ../run.py --max_output_len=20 --engine_dir=./trt_engine/gpt2/fp16/2-gpu/  --no_add_special_tokens
```

```
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
[TensorRT-LLM][INFO] MPI size: 2, rank: 1
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
[TensorRT-LLM][INFO] MPI size: 2, rank: 0
[TensorRT-LLM][WARNING] Device 1 peer access Device 0 is not available.
[TensorRT-LLM][INFO] Loaded engine size: 440 MiB
[TensorRT-LLM][INFO] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 583, GPU 727 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] Init cuDNN: CPU +2, GPU +10, now: CPU 585, GPU 737 (MiB)
[TensorRT-LLM][WARNING] TensorRT was linked against cuDNN 8.9.6 but loaded cuDNN 8.9.2
[TensorRT-LLM][WARNING] Device 0 peer access Device 1 is not available.
[TensorRT-LLM][INFO] Loaded engine size: 440 MiB
[TensorRT-LLM][INFO] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 583, GPU 727 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] Init cuDNN: CPU +2, GPU +10, now: CPU 585, GPU 737 (MiB)
[TensorRT-LLM][WARNING] TensorRT was linked against cuDNN 8.9.6 but loaded cuDNN 8.9.2
[TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +437, now: CPU 0, GPU 437 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +437, now: CPU 0, GPU 437 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 828, GPU 935 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 828, GPU 935 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] Init cuDNN: CPU +0, GPU +8, now: CPU 828, GPU 943 (MiB)
[TensorRT-LLM][WARNING] TensorRT was linked against cuDNN 8.9.6 but loaded cuDNN 8.9.2
[TensorRT-LLM][INFO] [MemUsageChange] Init cuDNN: CPU +0, GPU +8, now: CPU 828, GPU 943 (MiB)
[TensorRT-LLM][WARNING] TensorRT was linked against cuDNN 8.9.6 but loaded cuDNN 8.9.2
[TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 437 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 437 (MiB)
[TensorRT-LLM][INFO] Allocate 22007513088 bytes for k/v cache.
[TensorRT-LLM][INFO] Using 447744 tokens in paged KV cache.
[TensorRT-LLM][INFO] Allocate 22007513088 bytes for k/v cache.
[TensorRT-LLM][INFO] Using 447744 tokens in paged KV cache.
[TensorRT-LLM] TensorRT-LLM version: 0.8.0Input [Text 0]: "Born in north-east France, Soyer trained as a"
Output [Text 0 Beam 0]: " chef before moving to London in the early 1990s. He has since worked in restaurants in London,"
```

## 使用 GPT 生成摘要 

以下部分描述了如何运行一个 TensorRT-LLM GPT 模型来对 cnn_dailymail 数据集中的文章进行摘要。

对于每个摘要，脚本可以计算 ROUGE 分数，并使用 ROUGE-1 分数来验证。该脚本还可以使用 HF GPT 模型执行相同的摘要生成。

```
# Run the summarization task.
python3 ../summarize.py --engine_dir trt_engine/gpt2/fp16/1-gpu \
                        --hf_model_dir ./gpt2-medium/ \
                        --test_trt_llm \
                        --test_hf \
                        --batch_size 1 \
                        --check_accuracy \
                        --tensorrt_llm_rouge1_threshold=14 \
                        --no_add_special_tokens
```

#### ROUGE（Recall-Oriented Understudy for Gisting Evaluation）

ROUGE 是一组用于评估自动摘要和机器翻译软件的度量标准和软件包，在自然语言处理中广泛使用。
这些度量标准用于比较自动生成的摘要或翻译与参考摘要或翻译（由人类生成）之间的相似性。
ROUGE 度量标准的取值范围在 0 到 1 之间，较高的分数表示自动生成的摘要与参考摘要之间的相似性较高。

运行效果如下：

```
python3 ../summarize.py --engine_dir trt_engine/gpt2/fp16/1-gpu \
                    --hf_model_dir ./gpt2-medium/ \
                    --test_trt_llm \
                    --test_hf \
                    --batch_size 1 \
                    --check_accuracy \
                    --tensorrt_llm_rouge1_threshold=14 \
                    --no_add_special_tokens
[TensorRT-LLM] TensorRT-LLM version: 0.8.0[03/10/2024-09:29:11] [TRT-LLM] [I] Load tokenizer takes: 0.10835480690002441 sec
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
[TensorRT-LLM][INFO] MPI size: 1, rank: 0
[TensorRT-LLM][INFO] Loaded engine size: 777 MiB
[TensorRT-LLM][INFO] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 951, GPU 1065 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] Init cuDNN: CPU +2, GPU +10, now: CPU 953, GPU 1075 (MiB)
[TensorRT-LLM][WARNING] TensorRT was linked against cuDNN 8.9.6 but loaded cuDNN 8.9.2
[TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +774, now: CPU 0, GPU 774 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 981, GPU 1265 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] Init cuDNN: CPU +0, GPU +8, now: CPU 981, GPU 1273 (MiB)
[TensorRT-LLM][WARNING] TensorRT was linked against cuDNN 8.9.6 but loaded cuDNN 8.9.2
[TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 774 (MiB)
[TensorRT-LLM][INFO] Allocate 21705523200 bytes for k/v cache.
[TensorRT-LLM][INFO] Using 220800 tokens in paged KV cache.
[03/10/2024-09:29:12] [TRT-LLM] [I] Load engine takes: 0.9426615238189697 sec
The BetterTransformer implementation does not support padding during training, as the fused kernels do not support attention masks. Beware that passing padded batched data during training may result in unexpected outputs. Please refer to https://huggingface.co/docs/optimum/bettertransformer/overview for more details.
[03/10/2024-09:29:20] [TRT-LLM] [I] Load HF model takes: 8.465428829193115 sec
[03/10/2024-09:29:21] [TRT-LLM] [I] ---------------------------------------------------------
[03/10/2024-09:29:21] [TRT-LLM] [I] TensorRT-LLM Generated :
[03/10/2024-09:29:21] [TRT-LLM] [I]  Input : ['(CNN)James Best, best known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV\'s "The Dukes of Hazzard," died Monday after a brief illness. He was 88. Best died in hospice in Hickory, North Carolina, of complications from pneumonia, said Steve Latshaw, a longtime friend and Hollywood colleague. Although he\'d been a busy actor for decades in theater and in Hollywood, Best didn\'t become famous until 1979, when "The Dukes of Hazzard\'s" cornpone charms began beaming into millions of American homes almost every Friday night. For seven seasons, Best\'s Rosco P. Coltrane chased the moonshine-running Duke boys back and forth across the back roads of fictitious Hazzard County, Georgia, although his "hot pursuit" usually ended with him crashing his patrol car. Although Rosco was slow-witted and corrupt, Best gave him a childlike enthusiasm that got laughs and made him endearing. His character became known for his distinctive "kew-kew-kew" chuckle and for goofy catchphrases such as "cuff \'em and stuff \'em!" upon making an arrest. Among the most popular shows on TV in the early \'80s, "The Dukes of Hazzard" ran until 1985 and spawned TV movies, an animated series and video games. Several of Best\'s "Hazzard" co-stars paid tribute to the late actor on social media. "I laughed and learned more from Jimmie in one hour than from anyone else in a whole year," co-star John Schneider, who played Bo Duke, said on Twitter. "Give Uncle Jesse my love when you see him dear friend." "Jimmy Best was the most constantly creative person I have ever known," said Ben Jones, who played mechanic Cooter on the show, in a Facebook post. "Every minute of his long life was spent acting, writing, producing, painting, teaching, fishing, or involved in another of his life\'s many passions." Born Jewel Guy on July 26, 1926, in Powderly, Kentucky, Best was orphaned at 3 and adopted by Armen and Essa Best, who renamed him James and raised him in rural Indiana. Best served in the Army during World War II before launching his acting career. In the 1950s and 1960s, he accumulated scores of credits, playing a range of colorful supporting characters in such TV shows as "The Twilight Zone," "Bonanza," "The Andy Griffith Show" and "Gunsmoke." He later appeared in a handful of Burt Reynolds\' movies, including "Hooper" and "The End." But Best will always be best known for his "Hazzard" role, which lives on in reruns. "Jimmie was my teacher, mentor, close friend and collaborator for 26 years," Latshaw said. "I directed two of his feature films, including the recent \'Return of the Killer Shrews,\' a sequel he co-wrote and was quite proud of as he had made the first one more than 50 years earlier." People we\'ve lost in 2015 . CNN\'s Stella Chan contributed to this story.']
[03/10/2024-09:29:21] [TRT-LLM] [I]
 Reference : ['James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88 .\n"Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV .']
[03/10/2024-09:29:21] [TRT-LLM] [I]
 Output : [[' James Best died of pneumonia.']]
[03/10/2024-09:29:21] [TRT-LLM] [I] ---------------------------------------------------------
/usr/local/lib/python3.10/dist-packages/transformers/generation/configuration_utils.py:404: UserWarning: `do_sample` is set to `False`. However, `top_k` is set to `1` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_k`.
  warnings.warn(
/usr/local/lib/python3.10/dist-packages/transformers/generation/configuration_utils.py:430: UserWarning: `num_beams` is set to 1. However, `early_stopping` is set to `True` -- this flag is only used in beam-based generation modes. You should set `num_beams>1` or unset `early_stopping`.
  warnings.warn(
[03/10/2024-09:29:21] [TRT-LLM] [I] ---------------------------------------------------------
[03/10/2024-09:29:21] [TRT-LLM] [I] HF Generated :
[03/10/2024-09:29:21] [TRT-LLM] [I]  Input : ['(CNN)James Best, best known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV\'s "The Dukes of Hazzard," died Monday after a brief illness. He was 88. Best died in hospice in Hickory, North Carolina, of complications from pneumonia, said Steve Latshaw, a longtime friend and Hollywood colleague. Although he\'d been a busy actor for decades in theater and in Hollywood, Best didn\'t become famous until 1979, when "The Dukes of Hazzard\'s" cornpone charms began beaming into millions of American homes almost every Friday night. For seven seasons, Best\'s Rosco P. Coltrane chased the moonshine-running Duke boys back and forth across the back roads of fictitious Hazzard County, Georgia, although his "hot pursuit" usually ended with him crashing his patrol car. Although Rosco was slow-witted and corrupt, Best gave him a childlike enthusiasm that got laughs and made him endearing. His character became known for his distinctive "kew-kew-kew" chuckle and for goofy catchphrases such as "cuff \'em and stuff \'em!" upon making an arrest. Among the most popular shows on TV in the early \'80s, "The Dukes of Hazzard" ran until 1985 and spawned TV movies, an animated series and video games. Several of Best\'s "Hazzard" co-stars paid tribute to the late actor on social media. "I laughed and learned more from Jimmie in one hour than from anyone else in a whole year," co-star John Schneider, who played Bo Duke, said on Twitter. "Give Uncle Jesse my love when you see him dear friend." "Jimmy Best was the most constantly creative person I have ever known," said Ben Jones, who played mechanic Cooter on the show, in a Facebook post. "Every minute of his long life was spent acting, writing, producing, painting, teaching, fishing, or involved in another of his life\'s many passions." Born Jewel Guy on July 26, 1926, in Powderly, Kentucky, Best was orphaned at 3 and adopted by Armen and Essa Best, who renamed him James and raised him in rural Indiana. Best served in the Army during World War II before launching his acting career. In the 1950s and 1960s, he accumulated scores of credits, playing a range of colorful supporting characters in such TV shows as "The Twilight Zone," "Bonanza," "The Andy Griffith Show" and "Gunsmoke." He later appeared in a handful of Burt Reynolds\' movies, including "Hooper" and "The End." But Best will always be best known for his "Hazzard" role, which lives on in reruns. "Jimmie was my teacher, mentor, close friend and collaborator for 26 years," Latshaw said. "I directed two of his feature films, including the recent \'Return of the Killer Shrews,\' a sequel he co-wrote and was quite proud of as he had made the first one more than 50 years earlier." People we\'ve lost in 2015 . CNN\'s Stella Chan contributed to this story.']
[03/10/2024-09:29:21] [TRT-LLM] [I]
 Reference : ['James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88 .\n"Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV .']
[03/10/2024-09:29:21] [TRT-LLM] [I]
 Output : [[' James Best died of pneumonia.']]
[03/10/2024-09:29:21] [TRT-LLM] [I] ---------------------------------------------------------

[03/10/2024-09:29:40] [TRT-LLM] [I] TensorRT-LLM (total latency: 1.727743148803711 sec)
[03/10/2024-09:29:40] [TRT-LLM] [I] TensorRT-LLM (total output tokens: 672)
[03/10/2024-09:29:40] [TRT-LLM] [I] TensorRT-LLM (tokens per second: 388.9467022139794)
[03/10/2024-09:29:40] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[03/10/2024-09:29:40] [TRT-LLM] [I]   rouge1 : 21.95770615933248
[03/10/2024-09:29:40] [TRT-LLM] [I]   rouge2 : 6.127009262128831
[03/10/2024-09:29:40] [TRT-LLM] [I]   rougeL : 17.03914561308213
[03/10/2024-09:29:40] [TRT-LLM] [I]   rougeLsum : 18.885322760021552
[03/10/2024-09:29:40] [TRT-LLM] [I] Hugging Face (total latency: 11.438827991485596 sec)
[03/10/2024-09:29:40] [TRT-LLM] [I] HF beam 0 result
[03/10/2024-09:29:40] [TRT-LLM] [I]   rouge1 : 21.670397129686343
[03/10/2024-09:29:40] [TRT-LLM] [I]   rouge2 : 6.1022639415477835
[03/10/2024-09:29:40] [TRT-LLM] [I]   rougeL : 16.931744079691153
[03/10/2024-09:29:40] [TRT-LLM] [I]   rougeLsum : 18.92785002063369
```

输出的打印含义如下，其中 `Input` 和 `Reference` 来自数据集，`Output` 为模型的输出。

```
"---------------------------------------------------------")
logger.info("TensorRT-LLM Generated : ")
logger.info(f" Input : {datapoint[dataset_input_key]}")
logger.info(f"\n Reference : {datapoint[dataset_output_key]}")
logger.info(f"\n Output : {output}")
logger.info(
"---------------------------------------------------------")
```

执行多个 GPU

```
mpirun -n 2 --allow-run-as-root  python3 ../summarize.py --engine_dir trt_engine/gpt2/fp16/2-gpu --hf_model_dir ./gpt2-medium/ \
            --test_trt_llm --test_hf --batch_size 1 --check_accuracy \
            --tensorrt_llm_rouge1_threshold=14 --no_add_special_tokens
```

## Benchmark

具体参考文档 https://github.com/NVIDIA/TensorRT-LLM/tree/main/benchmarks/cpp

C++ Benchmark 需要在编译的时候使用参数

```
python3 scripts/build_wheel.py --clean --trt_root /usr/local/tensorrt --python_bindings --benchmarks
```

生成文件：
```
# ll cpp/build/benchmarks/
-rwxr-xr-x 1 root root 1569552 Mar 10 02:02 gptManagerBenchmark*
-rwxr-xr-x 1 root root 1404744 Mar 10 02:02 gptSessionBenchmark*
```

运行单 GPU Benchmark

```
./gptSessionBenchmark  --model gpt  --engine_dir  /path/to/gpt/trt_engine/gpt2/fp16/1-gpu/ --batch_size "1"  --input_output_len "60,100"
Benchmarking done. Iteration: 1521, duration: 60.04 sec.
Latencies: [42.37, 42.25, 42.23, 39.30, 38.78, 38.79, 38.81, 38.86, 38.77, 38.96 ... 40.48, 40.54, 42.38, 39.89, 39.69, 39.67, 39.64, 39.58, 39.58]
[BENCHMARK] batch_size 1 input_length 60 output_length 20 latency(ms) 39.47 tokensPerSec 506.69 gpu_peak_mem(gb) 23.06
```

```
./gptSessionBenchmark  --model gpt  --engine_dir  /path/to/gpt/trt_engine/gpt2/fp16/1-gpu/ --batch_size "1"  --input_output_len "60,100"
Benchmarking done. Iteration: 307, duration: 60.09 sec.
Latencies: [192.17, 192.18, 191.60, 192.02, 191.51, 191.53, 191.48, 190.82, 191.15, 190.77 ... 241.72, 242.39, 241.54, 242.12, 241.82, 241.44, 241.88, 242.31, 242.08]
[BENCHMARK] batch_size 1 input_length 60 output_length 100 latency(ms) 195.73 tokensPerSec 510.92 gpu_peak_mem(gb) 23.06
```

运行多 GPU Benchmark

```
#mpirun --allow-run-as-root -n 2 ./gptSessionBenchmark \
    --model gpt \
    --engine_dir "/path/to/gpt/trt_engine/gpt2/fp16/2-gpu/" \
    --batch_size "1" \
    --input_output_len "60,20"
Benchmarking done. Iteration: 1179, duration: 60.01 sec.
Latencies: [52.58, 50.16, 49.08, 49.56, 52.44, 49.52, 51.49, 50.99, 51.71, 50.14 ... 50.89, 51.45, 50.38, 50.82, 50.41, 50.54, 50.03, 50.23, 50.07]
[BENCHMARK] batch_size 1 input_length 60 output_length 20 latency(ms) 50.90 tokensPerSec 392.91 gpu_peak_mem(gb) 23.34
Benchmarking done. Iteration: 1179, duration: 60.01 sec.
Latencies: [52.69, 49.55, 49.67, 49.53, 52.11, 49.92, 51.50, 51.01, 51.70, 50.14 ... 51.01, 51.33, 50.18, 50.98, 50.64, 50.28, 50.16, 50.07, 50.18]
```