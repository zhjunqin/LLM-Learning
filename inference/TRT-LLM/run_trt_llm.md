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


## Profiling

使用 nsys profiling 

```
nsys profile --stats=true --sample=cpu --trace=cuda,cudnn,cublas,nvtx,osrt,oshmem \
    python ../run.py     --max_output_len 500     --max_input_length 32768    \
    --input_text "<s>[INST] <<SYS>>\nYour are an expert on C++ programing, help to answer user's question. \n<</SYS>>\n\nPlease give me the C++ style code to return all the Fibonacci numbers under 100. [/INST]"    \
    --engine_dir ./llama/int4/1-gpu/    \
    --tokenizer_dir /data/models/Llama-2-7b-chat-hf/
```

```
 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)   StdDev (ns)   Style                                                   Range        
 --------  ---------------  ---------  ---------  ---------  --------  ---------  -----------  --------  ----------------------------------------------------------------------------------------------------
     43.5       1563224408        443  3528723.3  2738453.0   2472373  226998610   11377453.2  PushPop   TensorRT:ExecutionContext::enqueue                          
      4.9        177388750      28795     6160.4     5754.0      4308     354623       2597.7  StartEnd  myelin-exec:myelinGraphExecute                              
      4.2        150865137        443   340553.4    29546.0     26048  137217100    6517915.9  PushPop   TensorRT:LLaMAForCausalLM/lm_head/PLUGIN_V2_Gemm_0          
      3.8        137647910        443   310717.6    14824.0     11928   80825689    4509232.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/0/attention/PLUGIN_V2_GPTAttention_0
      2.4         87509350         65  1346297.7  1335331.0   1312748    1725803      53272.9  StartEnd  myelinGraphDeserializeBinary                                
      1.2         43420237         65   668003.6   628715.0    587696    2056535     186583.3  StartEnd  myelin-exec:myelinGraphLoadPersistent                       
      0.9         32299461      57277      563.9      482.0       276      14194        326.6  StartEnd  myelin-exec:myelinTensorGetMemory                           
      0.6         21722615          5  4344523.0   264553.0    207762   20714650    9151254.5  PushPop   cuBLAS:cublasCreate_v2                                      
      0.5         18401219      28795      639.0      554.0       355      19290        344.9  StartEnd  myelin-exec:myelinGraphUnload                               
      0.4         15898291      28795      552.1      478.0       282      10871        349.6  StartEnd  myelin-exec:myelinGraphLoad                                 
      0.4         13339547        443    30111.8     8385.0      6479    8161392     392248.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/0/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.4         13238054      28795      459.7      398.0       214      11150        293.0  StartEnd  myelin-exec:myelinTensorGetInputMemoryBatch                 
      0.3         12225967      28795      424.6      359.0       198      11433        276.0  StartEnd  myelin-exec:myelinTensorGetOutputMemoryBatch                
      0.3         10893058        443    24589.3    22969.0     16458      55977       6372.3  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/vocab_embedding/CONSTANT_0...LLaMAForCausalLM/tr…
      0.3         10744843         65   165305.3   159040.0    147977     315746      25655.8  StartEnd  myelin-exec:myelinGraphUnloadPersistent                     
      0.3         10265519        443    23172.7    11411.0      9651    4988755     236465.0  PushPop   TensorRT:Reformatting CopyNode for Output Tensor 0 to LLaMAForCausalLM/lm_head/PLUGIN_V2_Gemm_0
      0.3         10069830        443    22731.0    21602.0     17262      45845       4130.4  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/ln_f/CONSTANT_1 + LLaMAForCausalLM/transformer/l…
      0.3          9475435      28795      329.1      220.0       168       9897        254.1  StartEnd  myelin-exec:myelinTensorSetInputMemoryBatch                 
      0.3          9398449      28795      326.4      258.0       156       9677        260.7  StartEnd  myelin-exec:myelinTensorSetOutputMemoryBatch                
      0.2          7653732        443    17277.0    13617.0     10745      47241       8125.5  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/7/input_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          7243246        443    16350.4    14627.0     12108      46149       4386.5  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/0/post_layernorm/CONSTANT_1 + LLaMAForCau…
      0.2          7100221        443    16027.6    14755.0     11177      39538       3813.9  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/28/post_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          7042504        443    15897.3    13847.0     11069     363006      16924.6  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/10/input_layernorm/CONSTANT_1 + LLaMAForC…
      0.2          7032282        443    15874.2    14532.0     11360      41551       3911.4  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/23/post_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          7031620        443    15872.7    14707.0     11099      57596       4077.5  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/31/input_layernorm/CONSTANT_1 + LLaMAForC…
      0.2          7029159        443    15867.2    14717.0     10914      34050       3764.7  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/30/post_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6971866        443    15737.8    14198.0     10932      38207       3929.9  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/19/post_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6964651        443    15721.6    14802.0     10788      31884       3654.0  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/29/post_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6961344        443    15714.1    14616.0     11154      37723       3769.0  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/29/input_layernorm/CONSTANT_1 + LLaMAForC…
      0.2          6945005        443    15677.2    14542.0     11023      42593       3695.4  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/23/input_layernorm/CONSTANT_1 + LLaMAForC…
      0.2          6922470        443    15626.3    14260.0     11090      31752       3828.6  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/31/post_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6919784        443    15620.3    14055.0     11299      45821       3897.7  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/17/post_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6918400        443    15617.2    14428.0     10612      47114       4109.1  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/24/input_layernorm/CONSTANT_1 + LLaMAForC…
      0.2          6899722        443    15575.0    14423.0     10832      31273       3615.7  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/24/post_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6889144        443    15551.1    14365.0     10981      34492       3610.8  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/25/post_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6875447        443    15520.2    14312.0     10656      45290       3859.2  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/30/input_layernorm/CONSTANT_1 + LLaMAForC…
      0.2          6850045        443    15462.9    13995.0     10918      32347       3982.6  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/27/post_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6840284        443    15440.8    14448.0     11019      31460       3628.6  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/26/input_layernorm/CONSTANT_1 + LLaMAForC…
      0.2          6803222        443    15357.2    14179.0     10801      31157       3683.9  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/25/input_layernorm/CONSTANT_1 + LLaMAForC…
      0.2          6801813        443    15354.0    13851.0     11046      38185       3962.6  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/1/input_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6800669        443    15351.4    13967.0     10528      45299       4046.6  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/21/post_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6795622        443    15340.0    13736.0     10896      28923       3902.9  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/10/post_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6793422        443    15335.0    14124.0     10773      33567       3662.3  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/28/input_layernorm/CONSTANT_1 + LLaMAForC…
      0.2          6787016        443    15320.6    13890.0     10999      40322       3856.1  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/20/input_layernorm/CONSTANT_1 + LLaMAForC…
      0.2          6782402        443    15310.2    13871.0     10833      34871       3882.1  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/17/input_layernorm/CONSTANT_1 + LLaMAForC…
      0.2          6778607        443    15301.6    13854.0     10886      41774       3841.1  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/9/input_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6773832        443    15290.8    13766.0     10962      32604       3509.1  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/22/post_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6772215        443    15287.2    13746.0     10923      31706       3679.8  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/20/post_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6761190        443    15262.3    13998.0     10598      32897       3651.0  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/21/input_layernorm/CONSTANT_1 + LLaMAForC…
      0.2          6760658        443    15261.1    13931.0     10873      38326       3906.5  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/12/input_layernorm/CONSTANT_1 + LLaMAForC…
      0.2          6758742        443    15256.8    13963.0     10399      31278       3812.4  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/27/input_layernorm/CONSTANT_1 + LLaMAForC…
      0.2          6754261        443    15246.6    13990.0     10973      28213       3425.5  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/22/input_layernorm/CONSTANT_1 + LLaMAForC…
      0.2          6747508        443    15231.4    14106.0     11013      29927       3394.1  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/8/input_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6740952        443    15216.6    13930.0     11025      34584       3476.3  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/13/post_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6735913        443    15205.2    14271.0     10650      30940       3412.4  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/26/post_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6730727        443    15193.5    13698.0     11033      32871       3878.9  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/16/post_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6717656        443    15164.0    13703.0     11231      29094       3671.0  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/19/input_layernorm/CONSTANT_1 + LLaMAForC…
      0.2          6712747        443    15152.9    13765.0     10989      32959       3750.6  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/9/post_layernorm/CONSTANT_1 + LLaMAForCau…
      0.2          6703559        443    15132.2    13757.0     10924      31295       3538.6  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/15/post_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6691839        443    15105.7    13566.0     11013      27180       3550.6  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/7/post_layernorm/CONSTANT_1 + LLaMAForCau…
      0.2          6683387        443    15086.7    13737.0     11014      28551       3551.0  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/12/post_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6682214        443    15084.0    13606.0     11067      38194       3908.0  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/13/input_layernorm/CONSTANT_1 + LLaMAForC…
      0.2          6670001        443    15056.4    13405.0     10507      38331       3875.9  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/11/post_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6669469        443    15055.2    13716.0     10729      30867       3633.7  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/8/post_layernorm/CONSTANT_1 + LLaMAForCau…
      0.2          6668221        443    15052.4    13770.0     10833      29001       3453.8  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/18/input_layernorm/CONSTANT_1 + LLaMAForC…
      0.2          6667347        443    15050.4    13663.0     11013      29735       3548.7  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/14/post_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6628747        443    14963.3    13669.0     10987      31795       3556.4  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/2/post_layernorm/CONSTANT_1 + LLaMAForCau…
      0.2          6627211        443    14959.8    13593.0     10723      29417       3527.9  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/18/post_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6614288        443    14930.7    13548.0     10905      27090       3401.3  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/14/input_layernorm/CONSTANT_1 + LLaMAForC…
      0.2          6602692        443    14904.5    13684.0     10887      31916       3595.8  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/16/input_layernorm/CONSTANT_1 + LLaMAForC…
      0.2          6589830        443    14875.5    13392.0     10621      29339       3659.1  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/11/input_layernorm/CONSTANT_1 + LLaMAForC…
      0.2          6589236        443    14874.1    13365.0     11240      31256       3744.2  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/3/post_layernorm/CONSTANT_1 + LLaMAForCau…
      0.2          6578310        443    14849.5    13491.0     10989      36740       3789.5  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/1/post_layernorm/CONSTANT_1 + LLaMAForCau…
      0.2          6573855        443    14839.4    13588.0     10840      28283       3284.2  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/15/input_layernorm/CONSTANT_1 + LLaMAForC…
      0.2          6561628        443    14811.8    13260.0     10888      41097       4053.2  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/4/input_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6547150        443    14779.1    13411.0     11035      35435       3752.8  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/2/input_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6535947        443    14753.8    13656.0     10734      27686       3303.0  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/3/input_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6499280        443    14671.1    13200.0     10968      35611       3887.4  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/4/post_layernorm/CONSTANT_1 + LLaMAForCau…
      0.2          6461273        443    14585.3    13241.0     10846      33529       3562.4  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/5/input_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6415736        443    14482.5    13299.0     11062      30645       3353.4  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/6/input_layernorm/CONSTANT_1 + LLaMAForCa…
      0.2          6374937        443    14390.4    13069.0     10746      31880       3384.1  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/6/post_layernorm/CONSTANT_1 + LLaMAForCau…
      0.2          6325511        443    14278.8    12938.0     10806      38728       3686.6  PushPop   TensorRT:{ForeignNode[LLaMAForCausalLM/transformer/layers/5/post_layernorm/CONSTANT_1 + LLaMAForCau…
      0.1          4888812        443    11035.7     9577.0      8421     456303      21325.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/8/attention/PLUGIN_V2_GPTAttention_0
      0.1          4840572        443    10926.8    10337.0      8971      75688       3586.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/1/attention/PLUGIN_V2_GPTAttention_0
      0.1          4561055        443    10295.8     9863.0      8564      49450       2437.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/2/attention/PLUGIN_V2_GPTAttention_0
      0.1          4557867        443    10288.6     9834.0      8568      43687       2448.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/13/attention/PLUGIN_V2_GPTAttention_0
      0.1          4554464        443    10281.0     9734.0      8659      46653       2637.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/17/attention/PLUGIN_V2_GPTAttention_0
      0.1          4534024        443    10234.8     9624.0      8383      44668       2614.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/11/attention/PLUGIN_V2_GPTAttention_0
      0.1          4532406        443    10231.2     9688.0      8484      58422       2911.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/24/attention/PLUGIN_V2_GPTAttention_0
      0.1          4525183        443    10214.9     9851.0      8551      48476       2387.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/3/attention/PLUGIN_V2_GPTAttention_0
      0.1          4516977        443    10196.3     9655.0      8578      46418       2696.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/4/attention/PLUGIN_V2_GPTAttention_0
      0.1          4506629        443    10173.0     9685.0      8545      50543       2508.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/9/attention/PLUGIN_V2_GPTAttention_0
      0.1          4504715        443    10168.7     9743.0      8451      44070       2253.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/15/attention/PLUGIN_V2_GPTAttention_0
      0.1          4500806        443    10159.8     9652.0      8349      44479       2422.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/19/attention/PLUGIN_V2_GPTAttention_0
      0.1          4492679        443    10141.5     9673.0      8542      43725       2254.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/29/attention/PLUGIN_V2_GPTAttention_0
      0.1          4487048        443    10128.8     9655.0      8385      45179       2312.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/21/attention/PLUGIN_V2_GPTAttention_0
      0.1          4486286        443    10127.1     9226.0      6823      30410       3025.0  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/0/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          4485942        443    10126.3     9675.0      8485      54456       2611.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/20/attention/PLUGIN_V2_GPTAttention_0
      0.1          4480572        443    10114.2     9673.0      8426      47646       2290.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/7/attention/PLUGIN_V2_GPTAttention_0
      0.1          4479796        443    10112.4     9745.0      8572      49859       2378.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/25/attention/PLUGIN_V2_GPTAttention_0
      0.1          4471988        443    10094.8     9621.0      8561      44369       2527.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/16/attention/PLUGIN_V2_GPTAttention_0
      0.1          4470785        443    10092.1     9677.0      8611      47334       2359.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/30/attention/PLUGIN_V2_GPTAttention_0
      0.1          4460878        443    10069.7     9647.0      8568      44171       2214.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/18/attention/PLUGIN_V2_GPTAttention_0
      0.1          4460619        443    10069.1     9668.0      8465      46106       2230.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/31/attention/PLUGIN_V2_GPTAttention_0
      0.1          4450751        443    10046.8     9536.0      8473      43108       2323.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/27/attention/PLUGIN_V2_GPTAttention_0
      0.1          4448037        443    10040.7     9623.0      8502      46148       2702.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/23/attention/PLUGIN_V2_GPTAttention_0
      0.1          4446683        443    10037.7     9654.0      8581      45086       2188.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/12/attention/PLUGIN_V2_GPTAttention_0
      0.1          4442771        443    10028.8     9552.0      8495      44531       2330.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/10/attention/PLUGIN_V2_GPTAttention_0
      0.1          4438781        443    10019.8     9667.0      8447      45324       2212.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/14/attention/PLUGIN_V2_GPTAttention_0
      0.1          4427092        443     9993.4     9649.0      8538      47593       2237.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/28/attention/PLUGIN_V2_GPTAttention_0
      0.1          4423623        443     9985.6     9596.0      8415      43480       2100.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/5/attention/PLUGIN_V2_GPTAttention_0
      0.1          4418191        443     9973.3     9539.0      8504      60205       2797.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/6/attention/PLUGIN_V2_GPTAttention_0
      0.1          4407623        443     9949.5     9565.0      8465      44384       2214.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/22/attention/PLUGIN_V2_GPTAttention_0
      0.1          4392295        443     9914.9     9475.0      8520      61257       2824.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/26/attention/PLUGIN_V2_GPTAttention_0
      0.1          2868311        443     6474.7     6211.0      4938      16712       1257.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/0/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2857010        443     6449.2     6311.0      4649      16078       1215.2  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/0/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tra…
      0.1          2801701        443     6324.4     6024.0      4924      45515       2236.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/0/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2774419        443     6262.8     5419.0      4419     300439      14041.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/11/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2718645        443     6136.9     5890.0      4748      23471       1351.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/1/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2705310        443     6106.8     5510.0      4396     199021       9235.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/13/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2684240        443     6059.2     5649.0      4631     120085       5538.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/13/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2672180        443     6032.0     5689.0      4710      72101       3328.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/0/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2669690        443     6026.4     5053.0      4033     353946      16595.7  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/10/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausal…
      0.1          2650166        443     5982.3     5742.0      4672      19915       1210.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/1/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2643103        443     5966.4     5779.0      4751      16084       1096.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/6/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2637130        443     5952.9     5649.0      4667      21094       1544.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/20/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2634874        443     5947.8     5741.0      4629      19803       1310.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/4/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2633849        443     5945.5     5717.0      4829      15192       1197.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/30/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2631135        443     5939.4     5762.0      4603      14254        950.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/7/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2626490        443     5928.9     5539.0      4491     131450       6024.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/17/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2621496        443     5917.6     5705.0      4662      21174       1324.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/2/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2616560        443     5906.5     5738.0      4660      15787       1072.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/2/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2612009        443     5896.2     5708.0      4804      15586       1062.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/3/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2611472        443     5895.0     5666.0      4647      15803       1287.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/22/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2609459        443     5890.4     5631.0      4652      20835       1501.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/20/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2607344        443     5885.7     5651.0      4741      16865       1120.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/31/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2606469        443     5883.7     5657.0      4740      16129       1089.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/11/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2606468        443     5883.7     5615.0      4615      14247       1263.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/24/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2605428        443     5881.3     5655.0      4603      15202       1119.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/29/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2604650        443     5879.6     5647.0      4581      14310       1077.4  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/0/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalLM…
      0.1          2599152        443     5867.2     5687.0      4678      11818        995.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/24/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2598598        443     5865.9     5656.0      4663      12863        997.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/4/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2594834        443     5857.4     5736.0      4713      13949        881.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/16/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2594154        443     5855.9     5668.0      4648      13168       1058.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/17/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2593304        443     5854.0     5613.0      4765      14818       1156.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/7/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2591696        443     5850.3     5665.0      4665      13658        931.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/3/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2591548        443     5850.0     5647.0      4700      15358       1070.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/28/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2589204        443     5844.7     5648.0      4587      16622       1172.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/9/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2588510        443     5843.1     5609.0      4545      25968       1376.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/0/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2587035        443     5839.8     5655.0      4651      19158       1125.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/21/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2586972        443     5839.7     5640.0      4650      19131       1130.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/15/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2585026        443     5835.3     5632.0      4642      35168       1661.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/9/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2583178        443     5831.1     5658.0      4662      14533        892.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/5/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2582272        443     5829.1     5595.0      4628      18707       1280.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/29/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2581451        443     5827.2     5631.0      4783      11710        897.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/19/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2579597        443     5823.0     5599.0      4625      17030       1176.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/17/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2578777        443     5821.2     5584.0      4640      15842       1206.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/16/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2577894        443     5819.2     5603.0      4673      14994       1134.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/25/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2577755        443     5818.9     5597.0      4645      17291       1218.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/13/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2576255        443     5815.5     5568.0      4673      18669       1289.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/30/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2574461        443     5811.4     5619.0      4625      14549        937.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/28/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2574071        443     5810.5     5605.0      4703      14182       1021.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/27/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2573508        443     5809.3     5571.0      4638      14426       1237.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/21/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2571595        443     5805.0     5597.0      4652      14242       1084.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/15/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2571354        443     5804.4     5600.0      4593      13753        990.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/23/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2571043        443     5803.7     5577.0      4697      15395       1071.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/27/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2570575        443     5802.7     5584.0      4612      18232       1278.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/8/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2570439        443     5802.3     5598.0      4657      13657       1038.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/26/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2569693        443     5800.7     5612.0      4686      16687       1055.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/26/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2569439        443     5800.1     5574.0      4640      15292       1031.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/10/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2568505        443     5798.0     5687.0      4705      11896        838.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/8/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2567405        443     5795.5     5573.0      4631      15487       1150.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/11/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2565865        443     5792.0     5669.0      4666      12254        813.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/18/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2564219        443     5788.3     5599.0      4633      22669       1246.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/24/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2562835        443     5785.2     5527.0      4677      16374       1240.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/25/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2562525        443     5784.5     5486.0      4601      18350       1369.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/1/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2561069        443     5781.2     5597.0      4618      13434       1108.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/30/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2559697        443     5778.1     5616.0      4790      14265        986.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/14/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2550982        443     5758.4     5457.0      4425      15908       1301.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/29/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2549849        443     5755.9     5572.0      4605      15312       1069.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/15/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2549034        443     5754.0     5556.0      4582      14412        995.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/31/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2547851        443     5751.4     5575.0      4605      14446       1074.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/23/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2546449        443     5748.2     5556.0      4676      14996       1017.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/10/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2545109        443     5745.2     5552.0      4646      17536       1168.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/5/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2544900        443     5744.7     5564.0      4617      14177        990.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/12/mlp/gate/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2544823        443     5744.5     5612.0      4618      13641        884.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/12/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2543676        443     5741.9     5524.0      4457      19360       1325.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/8/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2540509        443     5734.8     5499.0      4487      14820       1115.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/28/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2539437        443     5732.4     5466.0      4440      20188       1423.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/14/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2537324        443     5727.6     5548.0      4411      15914       1095.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/5/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2531748        443     5715.0     5535.0      4480      17805       1189.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/24/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2531661        443     5714.8     5426.0      4402      15310       1323.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/12/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2531453        443     5714.3     5541.0      4371      14817       1085.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/18/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2531445        443     5714.3     5564.0      4672      14387        967.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/22/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2531316        443     5714.0     5510.0      4503      16355       1124.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/2/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2531193        443     5713.8     5575.0      4653      13368        833.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/18/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2529769        443     5710.5     5570.0      4681      13172        864.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/14/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2527415        443     5705.2     5455.0      4449      19631       1256.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/14/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2526803        443     5703.8     5529.0      4667      15097        969.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/6/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2524282        443     5698.2     5485.0      4488      13771       1043.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/2/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2523700        443     5696.8     5481.0      4378      16025       1235.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/3/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2522492        443     5694.1     5522.0      4395      15862       1079.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/17/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2520640        443     5689.9     5525.0      4464      21080       1153.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/4/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2519327        443     5687.0     5502.0      4472      14650       1060.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/30/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2518956        443     5686.1     5515.0      4633      14562        976.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/19/attention/qkv/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2517724        443     5683.3     5407.0      4417      16485       1308.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/4/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2515968        443     5679.4     5462.0      4433      18467       1197.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/7/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2515507        443     5678.3     5487.0      4382      20896       1192.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/9/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2515472        443     5678.3     5536.0      4504      16144        932.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/16/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2514212        443     5675.4     5462.0      4441      13793       1120.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/13/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2513739        443     5674.4     5408.0      4459      17269       1363.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/17/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2511243        443     5668.7     5452.0      4475      15792       1022.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/1/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2510866        443     5667.9     5482.0      4469      16654       1061.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/29/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2509591        443     5665.0     5495.0      4441      12897        998.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/21/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2509286        443     5664.3     5522.0      4491      12078        873.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/7/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2508804        443     5663.2     5370.0      4518      17658       1328.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/5/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2506827        443     5658.8     5408.0      4420      13514       1104.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/10/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2505811        443     5656.5     5453.0      4425      15591       1164.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/19/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2505206        443     5655.1     5548.0      4416      11779        785.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/6/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2504190        443     5652.8     5406.0      4411      16811       1252.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/12/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2503815        443     5652.0     5438.0      4388      14475       1157.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/20/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2501824        443     5647.5     5373.0      4535      15324       1189.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/18/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2501753        443     5647.3     5483.0      4417      12384        880.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/8/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2500569        443     5644.6     5444.0      4441      12956       1118.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/22/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2499251        443     5641.7     5452.0      4418      14377       1051.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/1/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2499127        443     5641.4     5461.0      4456      14416        989.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/20/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2498321        443     5639.6     5490.0      4477      13618        861.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/31/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2497676        443     5638.1     5456.0      4423      13912       1027.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/3/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2497494        443     5637.7     5463.0      4422      15956       1026.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/25/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2496726        443     5636.0     5443.0      4417      10954        837.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/20/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2495075        443     5632.2     5375.0      4517      13591       1109.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/4/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2493694        443     5629.1     5471.0      4429      12705        977.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/10/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2493640        443     5629.0     5453.0      4434      14779        947.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/21/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2493125        443     5627.8     5495.0      4441      13329        892.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/22/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2493096        443     5627.8     5443.0      4469      15126       1007.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/28/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2492155        443     5625.6     5443.0      4459      14780       1086.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/31/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2490145        443     5621.1     5437.0      4487      15010        999.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/29/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2489923        443     5620.6     5479.0      4549      16471        964.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/25/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2489896        443     5620.5     5415.0      4452      12809       1028.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/16/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2488554        443     5617.5     5455.0      4424      14409        992.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/15/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2488536        443     5617.5     5410.0      4543      15764       1137.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/12/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2488222        443     5616.8     5358.0      4469      19474       1200.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/26/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2487189        443     5614.4     5426.0      4452      16352       1039.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/19/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2486546        443     5613.0     5441.0      4430      14195        999.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/23/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2486202        443     5612.2     5468.0      4451      15149       1025.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/5/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2485258        443     5610.1     5420.0      4487      13439       1004.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/15/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2484852        443     5609.1     5448.0      4518      13046        905.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/11/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2482237        443     5603.2     5428.0      4446      12646        942.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/16/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2481389        443     5601.3     5421.0      4437      16368       1057.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/23/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2481090        443     5600.7     5443.0      4452      12475        864.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/9/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2480120        443     5598.5     5375.0      4455      15843       1040.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/13/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2479607        443     5597.3     5427.0      4454      12701        931.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/25/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2476984        443     5591.4     5422.0      4465      12597        884.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/14/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2476849        443     5591.1     5428.0      4476      13386        974.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/30/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2474663        443     5586.1     5444.0      4447      12389        810.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/2/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2474602        443     5586.0     5390.0      4399      14361       1022.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/8/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2472801        443     5581.9     5383.0      4479      15065       1038.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/24/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2472792        443     5581.9     5404.0      4490      16609        981.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/9/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2470762        443     5577.3     5385.0      4441      12142        924.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/27/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2470443        443     5576.6     5419.0      4455      12187        848.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/3/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2470098        443     5575.8     5432.0      4473      13199        836.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/18/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2469462        443     5574.4     5420.0      4420      14291       1007.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/28/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2468653        443     5572.6     5364.0      4385      15859       1123.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/26/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2466404        443     5567.5     5414.0      4473      11489        823.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/6/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2465341        443     5565.1     5391.0      4434      14876        918.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/6/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2465113        443     5564.6     5372.0      4384      13816       1077.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/7/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2462053        443     5557.7     5423.0      4415      16079       1007.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/31/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2461032        443     5555.4     5388.0      4470      13110        894.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/19/mlp/proj/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2460552        443     5554.3     5388.0      4507      13973       1040.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/23/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2459932        443     5552.9     5327.0      4407      25620       1287.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/11/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2456486        443     5545.1     5383.0      4437      13263        927.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/22/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2455170        443     5542.1     5378.0      4459      14849        916.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/26/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2452775        443     5536.7     5387.0      4307      14041        969.1  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/31/ELEMENTWISE_SUM_1_output_0'nodeCastOp1__mye186…
      0.1          2447180        443     5524.1     5337.0      4422      16143       1078.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/10/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2445028        443     5519.3     5385.0      4509      11811        838.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/21/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2441127        443     5510.4     5352.0      4418      13271        990.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/27/mlp/fc/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2428708        443     5482.4     5167.0      4095      15883       1344.6  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/12/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausal…
      0.1          2423244        443     5470.1     5291.0      4428      16352        969.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/27/attention/dense/PLUGIN_V2_WeightOnlyQuantMatmul_0
      0.1          2411586        443     5443.8     5249.0      4177      20139       1122.1  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/3/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalLM…
      0.1          2407191        443     5433.8     5269.0      4325      12921        924.8  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/20/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausal…
      0.1          2396443        443     5409.6     5164.0      4058      19132       1236.2  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/19/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2392266        443     5400.1     5291.0      3776      11936        980.7  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/1/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tra…
      0.1          2392262        443     5400.1     5127.0      4126      15124       1292.7  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/10/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2386262        443     5386.6     5156.0      4132      17532       1105.5  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/28/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2382612        443     5378.4     5157.0      4083      16649       1157.0  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/23/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2378236        443     5368.5     5128.0      4109      36091       1694.5  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/17/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2373329        443     5357.4     5204.0      3885      11934        914.3  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/1/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2370098        443     5350.1     5198.0      3668      15358       1273.4  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/12/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tr…
      0.1          2363177        443     5334.5     5185.0      4111      12013        814.3  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/8/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2357353        443     5321.3     5147.0      3600      16197       1276.5  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/9/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tra…
      0.1          2355817        443     5317.9     5196.0      3643      15090       1180.6  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/21/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tr…
      0.1          2350618        443     5306.1     5100.0      4035      12429        983.7  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/17/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausal…
      0.1          2349871        443     5304.4     5191.0      3700      18902       1294.2  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/16/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tr…
      0.1          2349276        443     5303.1     5183.0      3610      15027       1023.8  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/5/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tra…
      0.1          2348113        443     5300.5     5172.0      3663      15725       1244.5  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/19/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tr…
      0.1          2346912        443     5297.8     5079.0      4148      13222       1024.9  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/27/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2345834        443     5295.3     5121.0      3647      19874       1457.6  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/15/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tr…
      0.1          2345203        443     5293.9     5130.0      3574      15797       1252.4  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/24/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tr…
      0.1          2344600        443     5292.6     5053.0      4090      19938       1298.7  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/16/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2342681        443     5288.2     5178.0      3563      14676       1196.9  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/8/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tra…
      0.1          2342493        443     5287.8     5097.0      4068      11612        856.9  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/30/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2341326        443     5285.2     5116.0      3666      19938       1277.9  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/3/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tra…
      0.1          2337509        443     5276.5     5148.0      3768      25397       1325.0  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/7/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tra…
      0.1          2337054        443     5275.5     5071.0      4122      12607       1027.8  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/19/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausal…
      0.1          2336549        443     5274.4     5107.0      3584      15683       1203.0  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/25/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tr…
      0.1          2335797        443     5272.7     5102.0      4010      13581        888.0  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/13/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2334861        443     5270.6     5152.0      3768      14211       1095.0  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/10/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tr…
      0.1          2334528        443     5269.8     5030.0      4106      15892       1166.9  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/22/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausal…
      0.1          2332207        443     5264.6     5204.0      3697      16775       1008.2  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/17/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tr…
      0.1          2332172        443     5264.5     5072.0      3655      14704       1302.6  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/23/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tr…
      0.1          2331779        443     5263.6     5021.0      4042      16472       1209.0  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/11/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2329730        443     5259.0     5044.0      4122      14771       1008.2  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/31/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausal…
      0.1          2328431        443     5256.1     5199.0      3597      15380       1080.6  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/31/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tr…
      0.1          2325985        443     5250.5     5171.0      3704      15494       1043.1  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/20/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tr…
      0.1          2323678        443     5245.3     5035.0      4141      12432        941.8  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/7/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalLM…
      0.1          2319073        443     5234.9     5051.0      4083      13961        973.7  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/15/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2318600        443     5233.9     5069.0      4000      14692       1020.6  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/25/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2317918        443     5232.3     4998.0      4012      21302       1274.1  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/1/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalLM…
      0.1          2317484        443     5231.3     5104.0      3661      16555       1058.5  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/2/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tra…
      0.1          2317149        443     5230.6     4950.0      4059      15154       1241.4  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/26/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausal…
      0.1          2315772        443     5227.5     5023.0      4041      13677       1014.8  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/5/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalLM…
      0.1          2315038        443     5225.8     5066.0      4008      12881        832.1  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/2/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalLM…
      0.1          2314642        443     5224.9     5181.0      3611      11919        858.2  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/13/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tr…
      0.1          2314301        443     5224.2     5015.0      4029      16033       1010.5  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/8/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalLM…
      0.1          2313560        443     5222.5     5141.0      3610      16606       1080.2  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/6/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tra…
      0.1          2311301        443     5217.4     5175.0      3636      17698        993.7  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/29/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tr…
      0.1          2311214        443     5217.2     5113.0      3640      13290       1067.0  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/26/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tr…
      0.1          2308228        443     5210.4     5022.0      3829      19905       1157.1  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/2/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2307224        443     5208.2     5023.0      3960      13802       1002.6  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/20/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2306910        443     5207.5     5108.0      3645      19789       1160.7  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/14/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tr…
      0.1          2306667        443     5206.9     5007.0      4055      15149        985.1  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/31/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2303657        443     5200.1     5180.0      3709      10355        777.7  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/18/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tr…
      0.1          2301545        443     5195.4     5010.0      3987      13642        962.7  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/12/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2301180        443     5194.5     4992.0      3996      13915       1031.6  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/9/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2300631        443     5193.3     5156.0      3608      14230        948.6  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/4/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tra…
      0.1          2299839        443     5191.5     4990.0      4004      14824       1031.2  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/14/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2296165        443     5183.2     4942.0      4041      15945       1094.4  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/13/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausal…
      0.1          2295839        443     5182.5     4992.0      3910      13153        995.1  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/24/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2294385        443     5179.2     4926.0      3896      13311       1116.8  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/24/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausal…
      0.1          2293230        443     5176.6     4985.0      3859      14298        977.2  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/21/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausal…
      0.1          2291316        443     5172.3     5017.0      4000      12581        874.3  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/3/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2290732        443     5171.0     5034.0      4081      10284        784.0  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/4/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalLM…
      0.1          2290385        443     5170.2     4943.0      3918      12251        940.5  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/9/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalLM…
      0.1          2290372        443     5170.1     5132.0      3688      11040        846.8  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/28/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tr…
      0.1          2287185        443     5162.9     4994.0      3924      12008        887.8  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/22/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2286780        443     5162.0     5146.0      3614      13581        852.9  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/30/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tr…
      0.1          2284721        443     5157.4     4989.0      3991      14744        970.9  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/29/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2282497        443     5152.4     4935.0      3969      20313       1111.5  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/26/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2282198        443     5151.7     5144.0      3732      12309        773.8  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/11/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tr…
      0.1          2281342        443     5149.8     4879.0      4001      15407       1196.4  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/4/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2280076        443     5146.9     5076.0      3606      11615        805.6  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/27/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tr…
      0.1          2278920        443     5144.3     4924.0      3881      17912       1163.5  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/5/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2276539        443     5138.9     5101.0      3695      14764        927.8  PushPop   TensorRT:PWN(PWN(PWN(LLaMAForCausalLM/transformer/layers/22/mlp/SIGMOID_0), PWN(LLaMAForCausalLM/tr…
      0.1          2276101        443     5137.9     4929.0      3973      11432        984.7  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/25/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausal…
      0.1          2274346        443     5134.0     4881.0      3917      36924       1811.4  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/30/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausal…
      0.1          2269447        443     5122.9     4890.0      3920      16135       1153.6  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/6/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2268275        443     5120.3     4943.0      3882      14308        988.6  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/18/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2265575        443     5114.2     4948.0      3971      10622        807.4  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/6/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalLM…
      0.1          2265154        443     5113.2     4916.0      3857      14938       1136.0  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/14/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausal…
      0.1          2263623        443     5109.8     4916.0      3888      14787       1085.2  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/16/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausal…
      0.1          2262416        443     5107.0     4944.0      3935      10730        838.9  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/29/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausal…
      0.1          2260574        443     5102.9     4970.0      3944      13520        846.7  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/23/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausal…
      0.1          2259093        443     5099.5     4919.0      3966      13500        874.5  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/7/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2257633        443     5096.2     4879.0      3966      14902       1082.9  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/28/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausal…
      0.1          2245987        443     5069.9     4868.0      3855      12609        980.2  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/11/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausal…
      0.1          2245505        443     5068.9     4863.0      3946      15503       1036.5  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/27/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausal…
      0.1          2238755        443     5053.6     4909.0      3858      13702        883.6  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/18/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausal…
      0.1          2238684        443     5053.5     4891.0      3886      19550       1109.6  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/21/post_layernorm/ELEMENTWISE_SUM_0LLaMAForCausalL…
      0.1          2214279        443     4998.4     4837.0      3854      14991        935.6  PushPop   myelin-exec:LLaMAForCausalLM/transformer/layers/15/input_layernorm/ELEMENTWISE_SUM_0LLaMAForCausal…
      0.1          2209447        443     4987.5     4851.0      3815      14679        933.5  PushPop   myelin-exec:LLaMAForCausalLM/transformer/ln_f/ELEMENTWISE_SUM_0LLaMAForCausalLM/transformer/ln_f/U…
      0.0          1102102        130     8477.7     8104.5      6538      15995       1488.6  StartEnd  myelin-exec:myelinGraphResolveShapes                        
      0.0           610478        443     1378.1     1215.0       809       8144        633.5  PushPop   TensorRT:Reformatting CopyNode for Network Input sequence_length
      0.0           262318        443      592.1      462.0       310       6107        585.7  PushPop   TensorRT:Reformatting CopyNode for Network Input cache_indirection
      0.0           260804        443      588.7      468.0       237       7209        506.4  PushPop   TensorRT:Reformatting CopyNode for Network Input host_request_types
      0.0           226814        443      512.0      415.0       224       4018        362.3  PushPop   TensorRT:Reformatting CopyNode for Network Input host_past_key_value_lengths
      0.0           224598        443      507.0      378.0       225       7340        570.6  PushPop   TensorRT:Reformatting CopyNode for Network Input context_lengths
      0.0           199852        443      451.1      360.0       215       9638        538.7  PushPop   TensorRT:Reformatting CopyNode for Network Input host_context_lengths
      0.0           190179        443      429.3      324.0       210       7064        532.9  PushPop   TensorRT:Reformatting CopyNode for Network Input host_sink_token_length
      0.0           172513        443      389.4      285.0       162       9820        557.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/28/mlp/proj/CONSTANT_1
      0.0           166726        443      376.4      277.0       160       7726        540.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/30/attention/dense/CONSTANT_1
      0.0           166150        443      375.1      275.0       168       6994        454.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/25/attention/dense/CONSTANT_0
      0.0           165645        443      373.9      194.0       171      21023       1023.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/17/attention/dense/CONSTANT_0
      0.0           165163        443      372.8      299.0       175       4487        365.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/0/attention/dense/CONSTANT_0
      0.0           165062        443      372.6      250.0       163      10271        618.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/31/mlp/proj/CONSTANT_0
      0.0           164654        443      371.7      278.0       169       2560        290.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/30/attention/dense/CONSTANT_0
      0.0           163910        443      370.0      276.0       173       7386        437.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/24/attention/dense/CONSTANT_0
      0.0           163375        443      368.8      204.0       159       6098        585.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/0/mlp/proj/CONSTANT_1
      0.0           163346        443      368.7      288.0       160       4753        341.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/22/mlp/proj/CONSTANT_1
      0.0           163042        443      368.0      280.0       161       7316        434.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/29/mlp/proj/CONSTANT_1
      0.0           162588        443      367.0      285.0       171       3421        293.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/26/attention/dense/CONSTANT_0
      0.0           162157        443      366.0      275.0       158       3237        318.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/31/mlp/proj/CONSTANT_1
      0.0           161802        443      365.2      277.0       164       2898        300.1  PushPop   TensorRT:LLaMAForCausalLM/lm_head/CONSTANT_0                
      0.0           160353        443      362.0      207.0       159       6773        417.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/17/mlp/proj/CONSTANT_1
      0.0           160075        443      361.3      188.0       169      19138        973.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/12/attention/dense/CONSTANT_0
      0.0           159898        443      360.9      275.0       168       2690        287.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/31/attention/dense/CONSTANT_0
      0.0           159293        443      359.6      267.0       160       2680        312.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/26/mlp/proj/CONSTANT_1
      0.0           158491        443      357.8      275.0       172       4914        325.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/27/attention/dense/CONSTANT_0
      0.0           158478        443      357.7      257.0       161       3904        334.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/12/mlp/proj/CONSTANT_1
      0.0           158207        443      357.1      293.0       162       2622        276.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/30/mlp/proj/CONSTANT_1
      0.0           156450        443      353.2      263.0       159       4193        321.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/20/mlp/proj/CONSTANT_1
      0.0           156167        443      352.5      268.0       159       5970        489.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/0/attention/dense/CONSTANT_1
      0.0           156025        443      352.2      268.0       169       6098        372.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/29/mlp/proj/CONSTANT_0
      0.0           155793        443      351.7      238.0       169       2338        280.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/23/attention/dense/CONSTANT_0
      0.0           155721        443      351.5      273.0       158       2580        271.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/31/attention/dense/CONSTANT_1
      0.0           155547        443      351.1      201.0       162       6600        381.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/27/mlp/proj/CONSTANT_1
      0.0           155510        443      351.0      208.0       157       4980        353.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/10/mlp/proj/CONSTANT_1
      0.0           155173        443      350.3      150.0       138      25515       1254.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/18/attention/qkv/CONSTANT_0
      0.0           154771        443      349.4      254.0       160       2729        294.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/24/mlp/proj/CONSTANT_1
      0.0           154336        443      348.4      180.0       158       8036        573.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/12/attention/dense/CONSTANT_1
      0.0           154296        443      348.3      233.0       159      13189        666.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/16/attention/dense/CONSTANT_1
      0.0           153885        443      347.4      240.0       160       3829        285.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/23/mlp/proj/CONSTANT_1
      0.0           153788        443      347.2      201.0       160       5003        352.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/16/mlp/proj/CONSTANT_1
      0.0           153760        443      347.1      203.0       160       7137        407.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/21/mlp/proj/CONSTANT_1
      0.0           153646        443      346.8      263.0       158       2584        271.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/26/attention/dense/CONSTANT_1
      0.0           153452        443      346.4      201.0       160       5470        359.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/18/mlp/proj/CONSTANT_1
      0.0           153401        443      346.3      205.0       160       7784        438.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/14/mlp/proj/CONSTANT_1
      0.0           153081        443      345.6      273.0       158       4447        310.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/26/mlp/proj/CONSTANT_0
      0.0           152747        443      344.8      188.0       168       5854        539.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/1/attention/dense/CONSTANT_0
      0.0           152642        443      344.6      268.0       162       4551        353.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/25/mlp/proj/CONSTANT_0
      0.0           152541        443      344.3      203.0       159       4602        327.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/8/mlp/proj/CONSTANT_1
      0.0           152479        443      344.2      211.0       161       2756        261.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/9/mlp/proj/CONSTANT_1
      0.0           152396        443      344.0      234.0       136      22677       1096.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/30/mlp/gate/CONSTANT_0
      0.0           152385        443      344.0      150.0       137      27785       1338.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/20/attention/qkv/CONSTANT_0
      0.0           152165        443      343.5      188.0       159       9758        547.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/17/mlp/proj/CONSTANT_0
      0.0           151837        443      342.7      273.0       171       2503        264.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/29/attention/dense/CONSTANT_0
      0.0           151547        443      342.1      184.0       157       5744        449.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/22/attention/dense/CONSTANT_1
      0.0           151141        443      341.2      217.0       159       3391        275.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/25/mlp/proj/CONSTANT_1
      0.0           150991        443      340.8      201.0       162       2717        297.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/24/mlp/proj/CONSTANT_0
      0.0           150261        443      339.2      264.0       158       4375        297.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/23/attention/dense/CONSTANT_1
      0.0           150186        443      339.0      183.0       157       7769        509.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/18/attention/dense/CONSTANT_1
      0.0           150137        443      338.9      266.0       158       4320        291.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/29/attention/dense/CONSTANT_1
      0.0           149639        443      337.8      267.0       159       2876        284.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/28/attention/dense/CONSTANT_1
      0.0           149438        443      337.3      261.0       166       2199        251.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/28/mlp/proj/CONSTANT_0
      0.0           148638        443      335.5      196.0       170       5545        342.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/28/attention/dense/CONSTANT_0
      0.0           148305        443      334.8      189.0       162       7480        434.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/21/mlp/proj/CONSTANT_0
      0.0           148052        443      334.2      200.0       159       6600        519.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/6/mlp/proj/CONSTANT_1
      0.0           148045        443      334.2      250.0       156       6388        374.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/25/attention/dense/CONSTANT_1
      0.0           147986        443      334.1      252.0       162       3967        293.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/27/mlp/proj/CONSTANT_0
      0.0           147885        443      333.8      192.0       159       5669        369.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/24/attention/dense/CONSTANT_1
      0.0           147520        443      333.0      241.0       161       2533        241.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/11/mlp/proj/CONSTANT_1
      0.0           147426        443      332.8      186.0       161      19782        959.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/3/mlp/proj/CONSTANT_0
      0.0           147088        443      332.0      187.0       172       7441        447.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/9/attention/dense/CONSTANT_0
      0.0           146958        443      331.7      189.0       169       6696        405.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/19/attention/dense/CONSTANT_0
      0.0           146897        443      331.6      233.0       133       7067        492.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/28/mlp/fc/CONSTANT_0
      0.0           146695        443      331.1      209.0       162       3132        268.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/7/mlp/proj/CONSTANT_1
      0.0           146589        443      330.9      190.0       169       4310        320.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/18/attention/dense/CONSTANT_0
      0.0           146325        443      330.3      185.0       158       7681        438.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/15/attention/dense/CONSTANT_1
      0.0           145899        443      329.3      204.0       158       4852        340.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/13/mlp/proj/CONSTANT_1
      0.0           145876        443      329.3      217.0       130       7579        566.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/29/mlp/gate/CONSTANT_1
      0.0           145850        443      329.2      192.0       161       2583        272.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/23/mlp/proj/CONSTANT_0
      0.0           145809        443      329.1      190.0       164       2972        280.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/21/attention/dense/CONSTANT_0
      0.0           145780        443      329.1      192.0       174       2554        272.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/11/attention/dense/CONSTANT_0
      0.0           145701        443      328.9      189.0       170       5991        372.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/14/attention/dense/CONSTANT_0
      0.0           145403        443      328.2      266.0       158       2474        237.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/27/attention/dense/CONSTANT_1
      0.0           145191        443      327.7      232.0       137       6683        547.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/0/attention/qkv/CONSTANT_0
      0.0           144981        443      327.3      187.0       158       2469        277.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/20/attention/dense/CONSTANT_1
      0.0           144970        443      327.2      192.0       161       4710        312.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/16/mlp/proj/CONSTANT_0
      0.0           144931        443      327.2      145.0       132       7709        571.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/26/mlp/fc/CONSTANT_1
      0.0           144771        443      326.8      187.0       166       3588        308.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/13/attention/dense/CONSTANT_0
      0.0           144621        443      326.5      201.0       160      10614        558.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/2/mlp/proj/CONSTANT_1
      0.0           144602        443      326.4      236.0       132      10609        590.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/28/mlp/fc/CONSTANT_1
      0.0           144165        443      325.4      262.0       162       2337        225.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/30/mlp/proj/CONSTANT_0
      0.0           144159        443      325.4      189.0       161       3397        299.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/11/mlp/proj/CONSTANT_0
      0.0           144127        443      325.3      188.0       163       6476        372.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/8/mlp/proj/CONSTANT_0
      0.0           144026        443      325.1      236.0       138       5106        356.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/28/mlp/gate/CONSTANT_0
      0.0           144021        443      325.1      155.0       132       6755        442.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/20/mlp/gate/CONSTANT_0
      0.0           144009        443      325.1      235.0       132       8848        471.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/28/mlp/gate/CONSTANT_1
      0.0           143801        443      324.6      244.0       138       6973        414.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/26/attention/qkv/CONSTANT_0
      0.0           143653        443      324.3      204.0       158       2432        252.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/15/mlp/proj/CONSTANT_1
      0.0           143543        443      324.0      190.0       162       5245        368.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/7/mlp/proj/CONSTANT_0
      0.0           143450        443      323.8      187.0       159       3967        297.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/22/mlp/proj/CONSTANT_0
      0.0           143354        443      323.6      188.0       160       4987        394.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/15/mlp/proj/CONSTANT_0
      0.0           143030        443      322.9      194.0       166       2548        274.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/9/mlp/proj/CONSTANT_0
      0.0           142811        443      322.4      189.0       170       5869        411.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/8/attention/dense/CONSTANT_0
      0.0           142642        443      322.0      190.0       161      11844        607.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/3/attention/dense/CONSTANT_0
      0.0           142388        443      321.4      243.0       159       4179        297.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/13/attention/dense/CONSTANT_1
      0.0           142348        443      321.3      201.0       158       5811        377.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/4/mlp/proj/CONSTANT_1
      0.0           142316        443      321.3      182.0       157       2527        273.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/14/attention/dense/CONSTANT_1
      0.0           142100        443      320.8      185.0       159       2361        248.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/9/attention/dense/CONSTANT_1
      0.0           141367        443      319.1      192.0       171       3375        293.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/19/mlp/proj/CONSTANT_0
      0.0           141276        443      318.9      205.0       161       2333        230.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/19/mlp/proj/CONSTANT_1
      0.0           141068        443      318.4      182.0       158       4344        312.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/19/attention/dense/CONSTANT_1
      0.0           140862        443      318.0      194.0       170       2356        233.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/10/attention/dense/CONSTANT_0
      0.0           140839        443      317.9      226.0       133       8523        506.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/31/attention/qkv/CONSTANT_1
      0.0           140415        443      317.0      179.0       158       3078        299.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/8/attention/dense/CONSTANT_1
      0.0           139990        443      316.0      192.0       164       2470        237.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/10/mlp/proj/CONSTANT_0
      0.0           139745        443      315.5      193.0       163       2711        257.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/12/mlp/proj/CONSTANT_0
      0.0           139646        443      315.2      156.0       135       7707        451.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/22/mlp/gate/CONSTANT_0
      0.0           139443        443      314.8      189.0       170       2238        240.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/16/attention/dense/CONSTANT_0
      0.0           139401        443      314.7      189.0       162       2342        258.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/20/attention/dense/CONSTANT_0
      0.0           138938        443      313.6      189.0       171       5135        397.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/0/mlp/proj/CONSTANT_0
      0.0           138537        443      312.7      181.0       158       2926        271.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/17/attention/dense/CONSTANT_1
      0.0           138506        443      312.7      181.0       158       6528        464.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/1/attention/dense/CONSTANT_1
      0.0           138353        443      312.3      238.0       136       4365        329.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/30/attention/qkv/CONSTANT_0
      0.0           137798        443      311.1      236.0       137       6197        371.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/29/mlp/gate/CONSTANT_0
      0.0           137654        443      310.7      228.0       132       4108        329.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/27/attention/qkv/CONSTANT_1
      0.0           137351        443      310.0      232.0       134       2760        302.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/24/attention/qkv/CONSTANT_0
      0.0           137331        443      310.0      190.0       163       3049        278.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/13/mlp/proj/CONSTANT_0
      0.0           137280        443      309.9      189.0       163       2310        230.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/18/mlp/proj/CONSTANT_0
      0.0           136964        443      309.2      183.0       160       2235        272.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/10/attention/dense/CONSTANT_1
      0.0           136896        443      309.0      237.0       136       3067        291.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/31/attention/qkv/CONSTANT_0
      0.0           136783        443      308.8      188.0       173       4140        355.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/5/attention/dense/CONSTANT_0
      0.0           136543        443      308.2      199.0       159       4348        322.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/3/mlp/proj/CONSTANT_1
      0.0           136452        443      308.0      148.0       136      14957        745.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/8/attention/qkv/CONSTANT_0
      0.0           136352        443      307.8      218.0       137       2993        301.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/26/mlp/gate/CONSTANT_0
      0.0           136338        443      307.8      189.0       164       2385        232.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/20/mlp/proj/CONSTANT_0
      0.0           136061        443      307.1      147.0       136       4503        421.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/16/mlp/gate/CONSTANT_0
      0.0           135949        443      306.9      236.0       158       2531        248.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/21/attention/dense/CONSTANT_1
      0.0           135713        443      306.3      241.0       131       4926        310.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/30/attention/qkv/CONSTANT_1
      0.0           135121        443      305.0      184.0       162       5626        397.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/5/mlp/proj/CONSTANT_0
      0.0           134999        443      304.7      144.0       132       7046        474.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/10/attention/qkv/CONSTANT_1
      0.0           134925        443      304.6      179.0       158       2774        273.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/7/attention/dense/CONSTANT_1
      0.0           134894        443      304.5      240.0       132       2921        266.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/22/mlp/gate/CONSTANT_1
      0.0           134527        443      303.7      188.0       168       1952        207.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/22/attention/dense/CONSTANT_0
      0.0           134369        443      303.3      188.0       162       2430        252.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/15/attention/dense/CONSTANT_0
      0.0           134021        443      302.5      233.0       136       5142        328.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/29/attention/qkv/CONSTANT_0
      0.0           133924        443      302.3      224.0       132       8313        481.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/31/mlp/fc/CONSTANT_0
      0.0           133872        443      302.2      148.0       136      11955        599.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/21/attention/qkv/CONSTANT_0
      0.0           133863        443      302.2      238.0       131       3027        288.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/25/mlp/fc/CONSTANT_0
      0.0           133318        443      300.9      147.0       130       7687        426.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/26/mlp/gate/CONSTANT_1
      0.0           133279        443      300.9      190.0       170       4287        316.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/7/attention/dense/CONSTANT_0
      0.0           133135        443      300.5      161.0       133       8532        494.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/23/mlp/fc/CONSTANT_0
      0.0           133069        443      300.4      238.0       132       3099        284.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/30/mlp/fc/CONSTANT_0
      0.0           132938        443      300.1      143.0       133      25618       1239.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/7/attention/qkv/CONSTANT_1
      0.0           132936        443      300.1      183.0       158       2862        250.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/11/attention/dense/CONSTANT_1
      0.0           132805        443      299.8      225.0       132       5776        381.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/31/mlp/gate/CONSTANT_1
      0.0           132122        443      298.2      143.0       133       8913        536.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/9/mlp/gate/CONSTANT_1
      0.0           132031        443      298.0      150.0       138       5624        530.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/0/mlp/gate/CONSTANT_0
      0.0           131656        443      297.2      215.0       137       8343        447.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/25/attention/qkv/CONSTANT_0
      0.0           131627        443      297.1      228.0       131       5088        351.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/29/attention/qkv/CONSTANT_1
      0.0           131590        443      297.0      232.0       130       5503        388.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/30/mlp/gate/CONSTANT_1
      0.0           131431        443      296.7      146.0       132       6637        389.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/24/mlp/gate/CONSTANT_1
      0.0           131344        443      296.5      216.0       133       2449        267.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/24/attention/qkv/CONSTANT_1
      0.0           131259        443      296.3      226.0       133       2514        274.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/29/mlp/fc/CONSTANT_0
      0.0           131071        443      295.9      148.0       135      12871        635.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/21/mlp/gate/CONSTANT_0
      0.0           131065        443      295.9      191.0       170       3304        272.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/2/attention/dense/CONSTANT_0
      0.0           130956        443      295.6      144.0       132       6832        436.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/11/mlp/gate/CONSTANT_1
      0.0           130923        443      295.5      228.0       139       2221        236.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/27/attention/qkv/CONSTANT_0
      0.0           130880        443      295.4      144.0       131       6996        472.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/11/mlp/fc/CONSTANT_0
      0.0           130874        443      295.4      178.0       159       2679        284.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/4/attention/dense/CONSTANT_1
      0.0           130720        443      295.1      233.0       130       2781        281.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/30/mlp/fc/CONSTANT_1
      0.0           130578        443      294.8      181.0       159       5238        373.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/2/attention/dense/CONSTANT_1
      0.0           130137        443      293.8      199.0       159       2764        268.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/5/mlp/proj/CONSTANT_1
      0.0           130009        443      293.5      142.0       133       7553        480.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/8/mlp/fc/CONSTANT_0
      0.0           129899        443      293.2      144.0       131       6600        463.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/19/attention/qkv/CONSTANT_1
      0.0           129813        443      293.0      147.0       137       2555        290.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/14/attention/qkv/CONSTANT_0
      0.0           129779        443      293.0      232.0       136       2254        246.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/31/mlp/gate/CONSTANT_0
      0.0           129725        443      292.8      146.0       130       7029        421.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/23/mlp/gate/CONSTANT_1
      0.0           129520        443      292.4      221.0       133       4941        315.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/29/mlp/fc/CONSTANT_1
      0.0           129506        443      292.3      148.0       139       5293        354.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/10/attention/qkv/CONSTANT_0
      0.0           129474        443      292.3      147.0       135       6027        413.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/8/mlp/gate/CONSTANT_0
      0.0           129395        443      292.1      184.0       131       7958        426.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/26/mlp/fc/CONSTANT_0
      0.0           129341        443      292.0      196.0       160       2976        279.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/1/mlp/proj/CONSTANT_1
      0.0           129104        443      291.4      144.0       131       6058        435.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/20/mlp/gate/CONSTANT_1
      0.0           128861        443      290.9      237.0       136       3004        252.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/25/mlp/gate/CONSTANT_0
      0.0           128860        443      290.9      150.0       131       5378        368.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/22/mlp/fc/CONSTANT_1
      0.0           128852        443      290.9      186.0       161       2447        229.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/14/mlp/proj/CONSTANT_0
      0.0           128631        443      290.4      149.0       139       4138        299.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/28/attention/qkv/CONSTANT_0
      0.0           128406        443      289.9      229.0       131       2410        255.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/20/mlp/fc/CONSTANT_0
      0.0           128250        443      289.5      163.0       132       3129        292.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/27/mlp/gate/CONSTANT_1
      0.0           128242        443      289.5      237.0       130       2781        268.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/26/attention/qkv/CONSTANT_1
      0.0           128190        443      289.4      147.0       139       8443        459.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/17/attention/qkv/CONSTANT_0
      0.0           128143        443      289.3      143.0       132       2943        304.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/23/attention/qkv/CONSTANT_1
      0.0           127998        443      288.9      224.0       132       4875        314.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/31/mlp/fc/CONSTANT_1
      0.0           127625        443      288.1      149.0       133       8522        458.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/18/attention/qkv/CONSTANT_1
      0.0           127493        443      287.8      187.0       164       2522        251.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/4/attention/dense/CONSTANT_0
      0.0           127426        443      287.6      143.0       132      11349        650.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/11/mlp/fc/CONSTANT_1
      0.0           127357        443      287.5      155.0       130       3511        319.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/13/mlp/fc/CONSTANT_0
      0.0           127187        443      287.1      221.0       133       2264        236.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/25/mlp/fc/CONSTANT_1
      0.0           127059        443      286.8      144.0       132       2706        268.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/20/mlp/fc/CONSTANT_1
      0.0           126869        443      286.4      226.0       132       2239        241.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/28/attention/qkv/CONSTANT_1
      0.0           126844        443      286.3      218.0       136       2148        226.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/23/mlp/gate/CONSTANT_0
      0.0           126826        443      286.3      148.0       132       3382        300.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/24/mlp/fc/CONSTANT_1
      0.0           126804        443      286.2      181.0       132       5474        342.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/13/mlp/fc/CONSTANT_1
      0.0           126745        443      286.1      167.0       130       4653        320.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/27/mlp/fc/CONSTANT_0
      0.0           126313        443      285.1      181.0       159       4703        330.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/3/attention/dense/CONSTANT_1
      0.0           126311        443      285.1      149.0       139       8495        454.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/11/attention/qkv/CONSTANT_0
      0.0           126024        443      284.5      145.0       131       2750        268.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/23/mlp/fc/CONSTANT_1
      0.0           126023        443      284.5      148.0       133       3690        305.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/12/mlp/gate/CONSTANT_0
      0.0           126021        443      284.5      146.0       131       8692        585.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/0/mlp/gate/CONSTANT_1
      0.0           125802        443      284.0      184.0       160       5275        332.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/2/mlp/proj/CONSTANT_0
      0.0           125746        443      283.9      178.0       158       2783        287.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/5/attention/dense/CONSTANT_1
      0.0           125722        443      283.8      193.0       136       5394        327.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/27/mlp/gate/CONSTANT_0
      0.0           125675        443      283.7      173.0       132       6413        421.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/0/attention/qkv/CONSTANT_1
      0.0           125194        443      282.6      148.0       136       2186        230.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/23/attention/qkv/CONSTANT_0
      0.0           125107        443      282.4      147.0       137       3175        298.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/12/attention/qkv/CONSTANT_0
      0.0           125047        443      282.3      143.0       131       5178        351.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/16/mlp/fc/CONSTANT_1
      0.0           125020        443      282.2      164.0       135       2441        250.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/24/mlp/gate/CONSTANT_0
      0.0           124955        443      282.1      145.0       132       4240        326.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/21/mlp/fc/CONSTANT_0
      0.0           124707        443      281.5      186.0       160       3111        275.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/1/mlp/proj/CONSTANT_0
      0.0           124524        443      281.1      143.0       132       9067        492.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/10/mlp/fc/CONSTANT_1
      0.0           124489        443      281.0      148.0       139       2677        271.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/16/attention/qkv/CONSTANT_0
      0.0           124375        443      280.8      145.0       131       5066        394.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/19/mlp/gate/CONSTANT_1
      0.0           124256        443      280.5      146.0       132       2334        250.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/25/mlp/gate/CONSTANT_1
      0.0           123789        443      279.4      148.0       137       5525        345.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/17/mlp/gate/CONSTANT_0
      0.0           123046        443      277.8      143.0       132       2994        294.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/15/mlp/gate/CONSTANT_1
      0.0           123042         65     1893.0     1677.0      1195       9350       1084.7  StartEnd  myelinBinaryGraphCreate                                     
      0.0           122720        443      277.0      146.0       133       5665        408.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/15/mlp/gate/CONSTANT_0
      0.0           122703        443      277.0      149.0       131       3247        271.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/25/attention/qkv/CONSTANT_1
      0.0           122656        443      276.9      147.0       138       2822        287.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/10/mlp/gate/CONSTANT_0
      0.0           122420        443      276.3      214.0       133       2826        243.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/24/mlp/fc/CONSTANT_0
      0.0           122393        443      276.3      148.0       135       2190        235.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/11/mlp/gate/CONSTANT_0
      0.0           122162        443      275.8      146.0       131       5222        445.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/0/mlp/fc/CONSTANT_1
      0.0           121976        443      275.3      147.0       132       2412        245.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/22/mlp/fc/CONSTANT_0
      0.0           121933        443      275.2      147.0       138       2295        257.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/22/attention/qkv/CONSTANT_0
      0.0           121854        443      275.1      143.0       131       3088        305.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/17/mlp/gate/CONSTANT_1
      0.0           121725        443      274.8      191.0       130       2527        236.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/12/mlp/fc/CONSTANT_0
      0.0           121709        443      274.7      147.0       136       3399        277.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/9/mlp/gate/CONSTANT_0
      0.0           121701        443      274.7      144.0       132       2343        255.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/21/attention/qkv/CONSTANT_1
      0.0           121335        443      273.9      149.0       131       2528        229.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/27/mlp/fc/CONSTANT_1
      0.0           121043        443      273.2      147.0       133       4197        295.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/7/mlp/gate/CONSTANT_0
      0.0           120882        443      272.9      146.0       131       2461        277.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/12/mlp/fc/CONSTANT_1
      0.0           120606        443      272.2      143.0       132       2752        295.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/9/mlp/fc/CONSTANT_0
      0.0           120392        443      271.8      147.0       135       2346        256.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/18/mlp/gate/CONSTANT_0
      0.0           120257        443      271.5      143.0       130       4888        329.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/18/mlp/fc/CONSTANT_0
      0.0           120028        443      270.9      147.0       136       4328        300.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/19/attention/qkv/CONSTANT_0
      0.0           119961        443      270.8      148.0       134       2507        286.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/2/attention/qkv/CONSTANT_0
      0.0           119736        443      270.3      143.0       131       5088        323.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/9/mlp/fc/CONSTANT_1
      0.0           119639        443      270.1      143.0       131       4328        339.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/14/mlp/fc/CONSTANT_1
      0.0           119311        443      269.3      143.0       131       5422        358.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/7/mlp/fc/CONSTANT_0
      0.0           119246        443      269.2      145.0       133       4696        303.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/21/mlp/fc/CONSTANT_1
      0.0           119231        443      269.1      147.0       137       4872        324.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/9/attention/qkv/CONSTANT_0
      0.0           119209        443      269.1      144.0       132       3077        278.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/16/mlp/fc/CONSTANT_0
      0.0           119155        443      269.0      144.0       131       2421        274.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/17/attention/qkv/CONSTANT_1
      0.0           119101        443      268.9      144.0       129       4734        323.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/17/mlp/fc/CONSTANT_0
      0.0           119034        443      268.7      143.0       131       3366        307.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/8/mlp/gate/CONSTANT_1
      0.0           119022        443      268.7      142.0       132       3207        285.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/13/attention/qkv/CONSTANT_1
      0.0           118961        443      268.5      143.0       131       5007        328.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/8/mlp/fc/CONSTANT_1
      0.0           118835        443      268.3      148.0       135       2548        241.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/13/attention/qkv/CONSTANT_0
      0.0           118790        443      268.1      143.0       130       2904        273.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/21/mlp/gate/CONSTANT_1
      0.0           118778        443      268.1      187.0       159       2405        223.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/6/attention/dense/CONSTANT_0
      0.0           118410        443      267.3      143.0       132       4143        318.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/11/attention/qkv/CONSTANT_1
      0.0           118195        443      266.8      145.0       131       2231        257.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/13/mlp/gate/CONSTANT_1
      0.0           118065        443      266.5      148.0       134       2699        251.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/13/mlp/gate/CONSTANT_0
      0.0           117498        443      265.2      147.0       133       3057        284.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/14/mlp/gate/CONSTANT_0
      0.0           117473        443      265.2      143.0       133       2328        237.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/10/mlp/fc/CONSTANT_0
      0.0           117454        443      265.1      143.0       131       2568        239.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/18/mlp/gate/CONSTANT_1
      0.0           117394        443      265.0      144.0       132       3021        282.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/18/mlp/fc/CONSTANT_1
      0.0           117388        443      265.0      143.0       132       4460        319.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/19/mlp/fc/CONSTANT_1
      0.0           117247        443      264.7      147.0       137       2129        222.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/19/mlp/gate/CONSTANT_0
      0.0           117143        443      264.4      174.0       159       2780        271.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/6/attention/dense/CONSTANT_1
      0.0           116314        443      262.6      144.0       131       2423        235.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/16/attention/qkv/CONSTANT_1
      0.0           116298        443      262.5      147.0       132       6108        407.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/0/mlp/fc/CONSTANT_0
      0.0           116260        443      262.4      144.0       132       2510        254.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/12/mlp/gate/CONSTANT_1
      0.0           116111        443      262.1      185.0       160       4859        285.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/6/mlp/proj/CONSTANT_0
      0.0           116003        443      261.9      143.0       132       2542        271.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/7/mlp/gate/CONSTANT_1
      0.0           115979        443      261.8      144.0       132       3404        288.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/8/attention/qkv/CONSTANT_1
      0.0           115978        443      261.8      183.0       158       2782        224.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/4/mlp/proj/CONSTANT_0
      0.0           115884        443      261.6      143.0       132       6725        370.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/20/attention/qkv/CONSTANT_1
      0.0           115845        443      261.5      147.0       132       2744        252.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/22/attention/qkv/CONSTANT_1
      0.0           115687        443      261.1      142.0       132       6983        472.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/6/mlp/fc/CONSTANT_1
      0.0           115623        443      261.0      146.0       132       2867        254.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/15/mlp/fc/CONSTANT_1
      0.0           115296        443      260.3      142.0       130       2507        257.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/17/mlp/fc/CONSTANT_1
      0.0           115256        443      260.2      147.0       134       5123        385.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/1/attention/qkv/CONSTANT_0
      0.0           115116        443      259.9      147.0       132       2689        261.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/15/mlp/fc/CONSTANT_0
      0.0           114942        443      259.5      142.0       131       4650        305.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/10/mlp/gate/CONSTANT_1
      0.0           113979        443      257.3      141.0       132       2415        265.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/15/attention/qkv/CONSTANT_1
      0.0           113968        443      257.3      143.0       131       2602        259.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/19/mlp/fc/CONSTANT_0
      0.0           113562        443      256.3      142.0       131       4186        409.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/1/attention/qkv/CONSTANT_1
      0.0           113253        443      255.7      145.0       135       9435        518.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/6/mlp/gate/CONSTANT_0
      0.0           113141        443      255.4      145.0       132       2887        232.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/14/attention/qkv/CONSTANT_1
      0.0           113059        443      255.2      146.0       132       2360        245.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/9/attention/qkv/CONSTANT_1
      0.0           112765        443      254.5      143.0       132       2458        245.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/12/attention/qkv/CONSTANT_1
      0.0           112175        443      253.2      141.0       130       5765        431.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/3/mlp/fc/CONSTANT_0
      0.0           112036        443      252.9      143.0       132       2301        221.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/16/mlp/gate/CONSTANT_1
      0.0           111823        443      252.4      148.0       138       2952        224.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/15/attention/qkv/CONSTANT_0
      0.0           111615        443      252.0      141.0       133       7966        450.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/3/attention/qkv/CONSTANT_1
      0.0           111539        443      251.8      141.0       131       2111        235.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/7/mlp/fc/CONSTANT_1
      0.0           111398        443      251.5      143.0       132       2742        250.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/3/mlp/fc/CONSTANT_1
      0.0           110087        443      248.5      146.0       133       9555        501.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/2/mlp/gate/CONSTANT_0
      0.0           109176        443      246.4      142.0       132       2622        291.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/3/mlp/gate/CONSTANT_1
      0.0           109060        443      246.2      148.0       140       2506        242.1  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/7/attention/qkv/CONSTANT_0
      0.0           108592        443      245.1      142.0       132       2218        222.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/14/mlp/fc/CONSTANT_0
      0.0           107829        443      243.4      141.0       132       6608        379.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/4/mlp/fc/CONSTANT_0
      0.0           107585        443      242.9      143.0       132       2708        274.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/5/mlp/gate/CONSTANT_1
      0.0           107479        443      242.6      142.0       131       6080        372.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/1/mlp/fc/CONSTANT_0
      0.0           107276        443      242.2      142.0       131       2234        230.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/14/mlp/gate/CONSTANT_1
      0.0           107259        443      242.1      146.0       135       3563        286.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/3/attention/qkv/CONSTANT_0
      0.0           107099        443      241.8      146.0       136       2248        247.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/4/attention/qkv/CONSTANT_0
      0.0           106936        443      241.4      142.0       132       3262        285.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/4/mlp/fc/CONSTANT_1
      0.0           106289        443      239.9      145.0       136       2606        258.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/3/mlp/gate/CONSTANT_0
      0.0           106232        443      239.8      146.0       133       2461        240.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/5/mlp/gate/CONSTANT_0
      0.0           106133        443      239.6      146.0       139       2631        258.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/5/attention/qkv/CONSTANT_0
      0.0           105971        443      239.2      144.0       131       3252        258.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/5/mlp/fc/CONSTANT_0
      0.0           105808        443      238.8      143.0       132       2453        229.4  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/2/attention/qkv/CONSTANT_1
      0.0           104871        443      236.7      142.0       131       2523        256.8  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/5/attention/qkv/CONSTANT_1
      0.0           104847        443      236.7      142.0       132       6059        351.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/2/mlp/fc/CONSTANT_0
      0.0           104823        443      236.6      146.0       137       3354        261.3  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/1/mlp/gate/CONSTANT_0
      0.0           104124        443      235.0      141.0       133       2363        217.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/4/mlp/gate/CONSTANT_1
      0.0           104073        443      234.9      141.0       130       4841        323.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/1/mlp/gate/CONSTANT_1
      0.0           103435        443      233.5      141.0       133       2163        239.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/6/mlp/fc/CONSTANT_0
      0.0           103180        443      232.9      142.0       132       2720        263.5  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/2/mlp/fc/CONSTANT_1
      0.0           101615        443      229.4      147.0       137       2790        251.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/6/attention/qkv/CONSTANT_0
      0.0           101262        443      228.6      141.0       131       2193        222.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/4/attention/qkv/CONSTANT_1
      0.0           101109        443      228.2      146.0       137       2486        247.9  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/4/mlp/gate/CONSTANT_0
      0.0            99525        443      224.7      142.0       131       2219        225.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/5/mlp/fc/CONSTANT_1
      0.0            98941        443      223.3      142.0       132       2221        205.2  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/2/mlp/gate/CONSTANT_1
      0.0            97876        443      220.9      142.0       133       2674        217.7  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/1/mlp/fc/CONSTANT_1
      0.0            97851        443      220.9      141.0       131       2342        208.6  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/6/mlp/gate/CONSTANT_1
      0.0            96825        443      218.6      141.0       131       2682        228.0  PushPop   TensorRT:LLaMAForCausalLM/transformer/layers/6/attention/qkv/CONSTANT_1

[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)    Min (ns)    Max (ns)   StdDev (ns)            Name
 --------  ---------------  ---------  ------------  ------------  ---------  ----------  ------------  ----------------------
     46.7      62335363587         60  1038922726.5  2005866923.0       8652  2014818337  1011739496.8  pthread_cond_wait
     31.3      41855070744        188   222633355.0       34951.0        210  8296093047  1275288958.6  epoll_wait
     13.0      17371629045          7  2481661292.1  2002022859.0   14268833  8688221095  2853193723.1  select
      3.7       5007041938         61    82082654.7   100122702.0       5806   100138737    36190053.1  poll
      3.7       5000699531         10   500069953.1   500069511.0  500063789   500074496        3226.4  pthread_cond_timedwait
      0.8       1123163944       6949      161629.6         644.0        230   591392147     8833553.7  read
      0.3        414082507       6069       68229.1        3061.0        237    16049185      314417.9  ioctl
      0.1        196425504         40     4910637.6     5223889.5       5051     9173398     2907389.8  pthread_mutex_lock
      0.1         90512594     281598         321.4          55.0         50       66328        1065.6  fgets
      0.0         45572332          9     5063592.4     5062584.0    5060861     5067705        2489.4  nanosleep
      0.0         30605582        674       45408.9        3230.0       1051    28116266     1082859.5  fopen
      0.0         28109576      11025        2549.6        2230.0        549       42751        1798.6  stat64
      0.0         14410274      10513        1370.7        1304.0        893       27879         576.0  lstat64
      0.0         13499424       2404        5615.4        4948.5       1134      150602        4502.7  open64
      0.0         12633115          1    12633115.0    12633115.0   12633115    12633115           0.0  fork
      0.0         11935336       3010        3965.2        3555.0       1002       27117        1716.7  openat
      0.0          5461050         38      143711.8       90749.5        495     1300226      229159.4  pthread_join
      0.0          5380308       4119        1306.2        1234.0        701       45468         780.3  fstat64
      0.0          4321076      89388          48.3          36.0         30       34711         222.8  pthread_cond_signal
      0.0          3538400        351       10080.9        1333.0        662      958481       73001.6  munmap
      0.0          2650271       1430        1853.3        1604.5       1012       22152        1298.1  socket
      0.0          2459924        173       14219.2       11803.0        558       32033        7647.1  write
      0.0          2261964        126       17952.1        5353.5       1175     1015696       90440.3  mmap64
      0.0          2038338         33       61767.8       61036.0      57580       75091        3563.6  sleep
      0.0          1940839         42       46210.5       41705.0      28860      120451       18988.6  pthread_create
      0.0          1386399        591        2345.9        1878.0        494       26191        1960.7  stat
      0.0          1297892          6      216315.3      207840.5      92213      329776       81909.8  pthread_rwlock_rdlock
      0.0          1177478        230        5119.5        1004.0        375      329311       30237.6  mmap
      0.0           912756        663        1376.7        1295.0        153       11106         843.3  fclose
      0.0           819500          5      163900.0      162013.0     157913      172988        5670.0  usleep
      0.0           662783         10       66278.3       32752.0      23212      332384       94322.1  sem_timedwait
      0.0           398686         23       17334.2       16771.0        824       32973       10106.5  writev
      0.0           397400         61        6514.8        3343.0       1636       36516        7225.4  open
      0.0           383858          4       95964.5       15410.0       9549      343489      165052.0  shmdt
      0.0           362354          4       90588.5        2125.0       1181      356923      177557.8  recv
      0.0           258509       1780         145.2          58.0         31       11863         690.2  pthread_cond_broadcast
      0.0           243282        167        1456.8        1283.0        331        6612         969.0  epoll_ctl
      0.0           194713         32        6084.8        4952.5       3548       11853        2343.5  fstatat
      0.0           176816         19        9306.1        7862.0        750       21049        7331.4  putc
      0.0           149637          7       21376.7       14214.0       3061       56093       20681.8  fopen64
      0.0           117174         32        3661.7        2997.5       1740        7375        1627.8  futex
      0.0           111543          4       27885.8       27466.5       5276       51334       21477.3  pthread_rwlock_wrlock
      0.0            96488          3       32162.7       10134.0       7586       78768       40381.5  connect
      0.0            89537        283         316.4         251.0        165        1334         180.7  fcntl
      0.0            71958         30        2398.6        1698.5        686        8725        1838.9  bind
      0.0            65816          5       13163.2        7761.0       5243       25130        8870.4  shutdown
      0.0            60574         17        3563.2        2142.0        411       12777        4182.3  fread
      0.0            58963          3       19654.3       13748.0       6901       38314       16518.4  send
      0.0            43480          9        4831.1        5705.0       2214        6959        1682.2  pipe
      0.0            43314          5        8662.8        7251.0       6125       14380        3294.2  socketpair
      0.0            43192          6        7198.7        4869.5       2261       17629        5995.6  sendmsg
      0.0            39825          6        6637.5        6266.0       3180       11368        3633.2  posix_fallocate
      0.0            37220        116         320.9         183.0         43        3247         429.7  sigaction
      0.0            34647          5        6929.4        6751.0       3214        9286        2476.4  shmget
      0.0            32781          8        4097.6        2817.5       1736       12486        3550.3  lstat
      0.0            30855        448          68.9          40.0         38         823          77.9  pthread_mutex_trylock
      0.0            29575         21        1408.3        1203.0        853        2724         552.6  listen
      0.0            28253          4        7063.3        6385.5       3789       11693        3542.6  shmat
      0.0            25367        538          47.2          48.0         37         175          11.1  flockfile
      0.0            17509          4        4377.3        3951.5       3183        6423        1459.9  process_vm_writev
      0.0            16665         19         877.1         401.0        121        5398        1221.4  fwrite
      0.0            16093          2        8046.5        8046.5       3795       12298        6012.5  accept
      0.0            14858         71         209.3          58.0         47        1703         307.3  fflush
      0.0            11859          1       11859.0       11859.0      11859       11859           0.0  pipe2
      0.0            11320         14         808.6         762.5        462        2105         410.3  fstat
      0.0             9623          6        1603.8        1360.5        786        3154         883.7  recvmsg
      0.0             6730          1        6730.0        6730.0       6730        6730           0.0  ftruncate
      0.0             6209          4        1552.3        1518.5       1111        2061         455.0  prctl
      0.0             4834          7         690.6         737.0        244        1172         340.8  shmctl
      0.0             4218         65          64.9          48.0         46         795          96.3  getc
      0.0             4207         10         420.7         376.0        215         923         202.9  dup
      0.0             3234          2        1617.0        1617.0        786        2448        1175.2  recvfrom
      0.0              883          3         294.3         298.0        226         359          66.6  signal

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls  Avg (ns)   Med (ns)   Min (ns)  Max (ns)   StdDev (ns)                 Name
 --------  ---------------  ---------  ---------  ---------  --------  ---------  -----------  ----------------------------------
     34.5        841064862        443  1898566.3  2013111.0      5952    2337568     362605.9  cudaEventSynchronize
     21.5        525048320        913   575080.3    10610.0      2484  516893906   17106396.2  cudaMemcpyAsync
     18.1        441970767      90929     4860.6     4396.0      2770    8701705      43406.3  cudaLaunchKernel
      7.8        191361283      43446     4404.6     4260.0      2572     353102       1985.1  cuLaunchKernel
      6.5        159408405        180   885602.3  1037185.5      2459    1061564     309492.7  cudaMalloc
      5.8        142388146        190   749411.3   731888.0       770   15541273    1189659.3  cudaFree
      2.3         57025633        532   107191.0     3035.5       862    9970341     545401.2  cudaMallocAsync_v11020
      1.6         38088056        164   232244.2   237939.5     79588     997350     135792.5  cuModuleLoadData
      0.5         11452835      29242      391.7      366.0       312      11230        160.5  cudaStreamIsCapturing_v10000
      0.4          9854520         66   149310.9   141451.5    125734     463988      42602.4  cuModuleUnload
      0.1          2746373         32    85824.2     3209.0      2442    2419917     427291.0  cudaStreamCreateWithFlags
      0.1          2626061         65    40400.9     7708.0      6498    1440121     181445.5  cuStreamCreate
      0.1          2493419          6   415569.8    34126.0      7298    1724839     694475.3  cudaFreeHost
      0.1          2471425          6   411904.2    38790.0      5538    1244762     596800.2  cudaHostAlloc
      0.1          1786924        886     2016.8     1852.0      1398       7600        630.7  cudaEventRecord
      0.1          1696163          5   339232.6   315744.0    264004     514895     101248.2  cudaGetDeviceProperties_v2_v12000
      0.1          1447000        532     2719.9     1931.0       733     375447      16215.5  cudaFreeAsync_v11020
      0.0          1071098       1334      802.9      480.0       345      18521        962.6  cudaEventCreateWithFlags
      0.0           717550         34    21104.4    19549.5     13650      32962       4855.9  cudaMemcpy
      0.0           640724        177     3619.9     2517.0       167      50406       5814.0  cudaMemsetAsync
      0.0           593988       2349      252.9      188.0       113       6625        276.0  cuGetProcAddress_v2
      0.0           572513        154     3717.6     1985.5       767      27367       3314.0  cudaDeviceSynchronize
      0.0           534292       1334      400.5      342.0       282       7204        279.7  cudaEventDestroy
      0.0           463861         65     7136.3     4138.0      3046     131427      16864.6  cuStreamDestroy_v2
      0.0           331236          9    36804.0    37808.0      9584      79425      21771.2  cudaMemGetInfo
      0.0           176857         36     4912.7     3465.5      2824      17947       3666.9  cudaStreamDestroy
      0.0            91690          3    30563.3     1517.0       264      89909      51398.7  cudaDeviceGetDefaultMemPool_v11020
      0.0            77017         65     1184.9     1161.0       990       2064        143.9  cuEventCreate
      0.0            69111         11     6282.8     5166.0       773      15477       4591.8  cudaStreamSynchronize
      0.0            59774          1    59774.0    59774.0     59774      59774          0.0  cudaStreamCreateWithPriority
      0.0            57601          3    19200.3    18504.0     15087      24010       4502.1  cudaStreamCreate
      0.0            43166        130      332.0      383.5       161        731        150.4  cuCtxSetCurrent
      0.0            41697         65      641.5      645.0       398       1036        175.4  cuEventDestroy_v2
      0.0            11171          6     1861.8     1520.0       977       2956        845.6  cuInit
      0.0             8391          1     8391.0     8391.0      8391       8391          0.0  cudaMemPoolSetAttribute_v11020
      0.0             6047          8      755.9      280.5       174       4150       1374.5  cuModuleGetLoadingMode
      0.0             2401          2     1200.5     1200.5       266       2135       1321.6  cudaMemPoolGetAttribute_v11020

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                                                  Name                     
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     76.6       1789724858      70720   25307.2   28000.0     11392     31328       6837.2  void tensorrt_llm::kernels::weight_only_batched_gemv_wrapper<__half, (tensorrt_llm::kernels::Weight…
      9.0        209467142      14144   14809.6   14879.0     10176     19584       2254.7  void tensorrt_llm::kernels::mmha::masked_multihead_attention_kernel<unsigned short, unsigned short,…
      6.6        153278103        443  346000.2  345823.0    338079    354271       1826.2  void cutlass::Kernel<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)
      4.1         95331183      27846    3423.5    3424.0      2912      3872        113.4  __myl_bb1_1_AddCasMulMeaAddSqrDivMulCasMul_com_vec                       
      1.3         29279850      14176    2065.5    2080.0      1824      7040        243.1  generatedNativePointwise                                                 
      0.9         20700270        443   46727.5   46719.0     29503     49343        950.2  void tensorrt_llm::kernels::batchApplyPenalty<float>(T1 *, const T1 *, int *, const int *, const fl…
      0.7         17350261        160  108439.1  118799.5     47775    135967      30928.8  void cutlass::Kernel<cutlass::gemm::kernel::GemmFpAIntB<cutlass::gemm::threadblock::DqMmaMultistage…
      0.3          5996179        442   13566.0   13552.0     12865     14368        178.3  __myl_bb1_1_GatCasMulMeaAddSqrDivMulCasMul_com                           
      0.1          2380637        443    5373.9    5376.0      5088      5760         79.5  void splitKreduce_kernel<(int)32, (int)16, int, __half, __half, float, __half, (bool)1, (bool)0, (b…
      0.1          2228791        443    5031.1    4992.0      4768      5568         89.8  void tensorrt_llm::kernels::topKStage2Sampling<float, (int)128, (int)8>(const int *, T1 *, int **, …
      0.1          1762041        443    3977.5    3968.0      3744      4288         62.8  void tensorrt_llm::kernels::topKStage1<float, (int)128, (int)8>(const T1 *, T1 *, int *, T1 *, cons…
      0.1          1501113        443    3388.5    3360.0      3232      3680         77.8  tensorrt_llm::kernels::copyNextStepIds(int *, int **, const int *, const int *, int, int, int)
      0.1          1239998        443    2799.1    2784.0      2656      3168         57.3  tensorrt_llm::kernels::lengthCriterion(tensorrt_llm::kernels::FinishedState *, int *, const unsigne…
      0.1          1223100        443    2760.9    2752.0      2592      2976         44.1  void tensorrt_llm::kernels::addBiasEndMask<float>(T1 *, const T1 *, const int *, const tensorrt_llm…
      0.1          1222142        442    2765.0    2752.0      2592      2944         45.8  __myl_bb1_2_AddCasMulMea_com_vec                                         
      0.0           992925        445    2231.3    2240.0      1824      2464         45.8  void cub::CUB_200200_700_800_860_890_900_NS::DeviceScanKernel<cub::CUB_200200_700_800_860_890_900_N…
      0.0           967705        442    2189.4    2176.0      2048      2336         42.7  __myl_bb1_1_AddSqrDivMulCasMulSubGat_com                                 
      0.0           726108        443    1639.1    1632.0      1472      1792         30.9  void genericReformat::copyVectorizedKernel<double, __half, float, (bool)1, (bool)0, (int)1>(unsigne…
      0.0           584861        444    1317.3    1312.0      1184      1408         25.3  void tensorrt_llm::runtime::kernels::<unnamed>::add<int>(T1 *, unsigned long, T1)
      0.0           476763        445    1071.4    1056.0       992      1152         22.3  void cub::CUB_200200_700_800_860_890_900_NS::DeviceScanInitKernel<cub::CUB_200200_700_800_860_890_9…
      0.0           476734        447    1066.5    1056.0       992      1312         24.8  void tensorrt_llm::runtime::kernels::<unnamed>::fill<int>(T1 *, unsigned long, T1)
      0.0           298367         32    9324.0    9264.0      9183      9984        160.5  void tensorrt_llm::kernels::applyBiasRopeUpdateKVCache<__half, __half, (int)128, (bool)0, (bool)1, …
      0.0           263680         63    4185.4    4160.0      4000      4448        109.8  __myl_bb1_1_AddCasMulMeaAddSqrDivMulCasMul_vec                           
      0.0           192798         32    6024.9    6016.0      5983      6336         60.9  fmha_v2_flash_attention_fp16_64_32_S_128_causal_sm86_kernel_nl           
      0.0           112191         64    1753.0    1728.0      1536      2048        175.4  void tensorrt_llm::kernels::computeSeqOffsets<(int)256>(int *, const int *, int)
      0.0            39904         32    1247.0    1248.0      1216      1280         12.8  tensorrt_llm::kernels::computePaddingOffsets(int *, const int *, int)    
      0.0            14464          1   14464.0   14464.0     14464     14464          0.0  __myl_bb1_1_GatCasMulMeaAddSqrDivMulCasMul                               
      0.0             2816          1    2816.0    2816.0      2816      2816          0.0  __myl_bb1_2_AddCasMulMea_vec                                             
      0.0             2336          2    1168.0    1168.0      1088      1248        113.1  void tensorrt_llm::kernels::scatterDecodingParamsKernel<float>(const T1 *, T1 *, const int *, int)
      0.0             2080          2    1040.0    1040.0      1024      1056         22.6  tensorrt_llm::kernels::curandInitialize(curandStateXORWOW *, const int *, int, unsigned long)
      0.0             2048          1    2048.0    2048.0      2048      2048          0.0  __myl_bb1_1_AddSqrDivMulCasMulSubGat                                     
      0.0             1632          1    1632.0    1632.0      1632      1632          0.0  tensorrt_llm::runtime::kernels::<unnamed>::copyPackedInputToOutput(int *, const int *, const int *,…
      0.0             1280          1    1280.0    1280.0      1280      1280          0.0  void tensorrt_llm::runtime::kernels::<unnamed>::tileTensor<int>(T1 *, const T1 *, unsigned int, uns…
      0.0             1184          1    1184.0    1184.0      1184      1184          0.0  tensorrt_llm::layers::set_topp_runtime_args(int, unsigned int, unsigned int *, int, float, float *,…
      0.0             1152          1    1152.0    1152.0      1152      1152          0.0  void tensorrt_llm::layers::setupTopKRuntimeArgs<(unsigned int)1024>(int, unsigned int, unsigned int…
      0.0             1120          1    1120.0    1120.0      1120      1120          0.0  void tensorrt_llm::kernels::scatterDecodingParamsKernel<int>(const T1 *, T1 *, const int *, int)

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)  Min (ns)  Max (ns)   StdDev (ns)            Operation
 --------  ---------------  -----  ---------  --------  --------  ---------  -----------  ------------------------------
     99.8        516958436    457  1131200.1     544.0       416  516705446   24170435.3  [CUDA memcpy Host-to-Device]
      0.1           591971    446     1327.3    1312.0      1120       1440         28.4  [CUDA memcpy Device-to-Device]
      0.0           186337    175     1064.8    1120.0       384       1952        269.9  [CUDA memset]
      0.0            62048     44     1410.2    1376.0      1248       1728        113.1  [CUDA memcpy Device-to-Host]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)            Operation
 ----------  -----  --------  --------  --------  --------  -----------  ------------------------------
   3766.791    457     8.242     0.003     0.000  3765.654      176.150  [CUDA memcpy Host-to-Device]
      0.678    175     0.004     0.001     0.000     0.512        0.039  [CUDA memset]
      0.004    446     0.000     0.000     0.000     0.002        0.000  [CUDA memcpy Device-to-Device]
      0.002     44     0.000     0.000     0.000     0.002        0.000  [CUDA memcpy Device-to-Host]

```
