# MLC 介绍

Machine Learning Compilation for Large Language Models(MLC-LLM) 是一种高性能通用部署解决方案，它允许使用编译器加速的本地 API 进行任何大型语言模型的本地部署。该项目的使命是通过使用机器学习编译技术(TVM)，使每个人都能够在自己的设备上本地开发、优化和部署 AI 模型。”

## 项目概览

MLC-LLM 包含三个独立的子模块：模型定义、模型编译和运行时。

![](./assets/project_overview.png)

- Python 中的模型定义（Model definition）
  - MLC 提供了各种预定义的架构，例如 Llama（例如 Llama2、Vicuna、OpenLlama、Wizard）、GPT-NeoX（例如 RedPajama、Dolly）、RNNs（例如 RWKV）和 GPT-J（例如 MOSS）。模型开发人员可以仅使用纯Python 定义模型，而无需接触代码生成和运行时。
- Python 中的模型编译（Model compilation）
  - 模型通过 TVM Unity 编译器进行编译，编译配置在纯 Python 中进行。MLC LLM 会对基于 Python 的模型进行量化和导出，生成模型库和量化模型权重。可以使用纯 Python 开发量化和优化算法，以压缩和加速特定的 LLM。
- 平台本地运行时（Platform-native runtimes）
  - 在每个平台上提供了 MLC-Chat 的变体：命令行的 C++ 版本、Web 的 Javascript 版本、iOS 的 Swift 版本和 Android 的 Java 版本，可通过 JSON 进行配置。应用程序开发人员只需熟悉平台本地运行时，即可将 MLC 编译的 LLM 集成到其项目中。


## 模型准备流程

MLC 运行模型所需的三个要素：
- 模型库(model lib)：模型库指的是可执行库，用于执行特定的模型架构。在 Linux 和 M-chip macOS 上，这些库的后缀为 .so；在 intel macOS 上，后缀为 .dylib ；在 Windows 上，库文件以 .dll 结尾；在 Web 浏览器上，库的后缀是 .wasm。
- 模型权重(model weights)：模型权重是一个包含语言模型的量化神经网络权重以及分词器配置的文件夹。
- Chat 配置：配置包括允许自定义参数（如 temperature，top_p 和 system prompt）的设置。默认的配置通常位于与模型权重在相同的目录中。

1. 下载原始模型
```
git clone https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1
```
2. 模型权重量化转换
```
# 权重转换
mlc_llm convert_weight ./dist/models/RedPajama-INCITE-Instruct-3B-v1/ \    
    --quantization q4f16_1 \     # 4bit量化
    -o dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC

# 转换后的权重文件
-rw-r--r-- 1 root root   206009 Mar 12 15:29 ndarray-cache-b16.json
-rw-r--r-- 1 root root   206457 Mar 12 15:29 ndarray-cache.json
-rw-r--r-- 1 root root 64552960 Mar 12 15:29 params_shard_0.bin
-rw-r--r-- 1 root root 22855680 Mar 12 15:29 params_shard_1.bin
。。。省略
-rw-r--r-- 1 root root 22830080 Mar 12 15:29 params_shard_50.bin
-rw-r--r-- 1 root root  2113738 Mar 12 14:52 tokenizer.json
-rw-r--r-- 1 root root      237 Mar 12 14:52 tokenizer_config.json
```

3. 模型 Chat 配置生成

```
# 生成 chat 配置
mlc_llm gen_config ./dist/models/RedPajama-INCITE-Instruct-3B-v1/ \    
    --quantization q4f16_1 \
    --conv-template redpajama_chat \    
    -o dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC/

# 查看生成的 Chat 配置
# cat dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC/mlc-chat-config.json
{
  "model_type": "gpt_neox",
  "quantization": "q4f16_1",
  "model_config": {
    "use_parallel_residual": false,
    "hidden_size": 2560,
    "intermediate_size": 10240,
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "layer_norm_eps": 1e-05,
    "vocab_size": 50432,
    "rotary_pct": 1.0,
    "position_embedding_base": 10000,
    "context_window_size": 2048,
    "head_dim": 80,
    "prefill_chunk_size": 2048,
    "tensor_parallel_shards": 1,
    "ffn_out_dtype": "float32",
    "max_batch_size": 80
  },
  "vocab_size": 50432, # 词表大小
  "context_window_size": 2048,
  "sliding_window_size": -1,
  "prefill_chunk_size": 2048,
  "attention_sink_size": -1,
  "tensor_parallel_shards": 1,
  "mean_gen_len": 128,
  "max_gen_len": 512,
  "shift_fill_factor": 0.3,
  "temperature": 0.7, # 解码 temperature 配置
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "repetition_penalty": 1.0,
  "top_p": 0.95, # TOP_P 配置项
  "conv_template": "redpajama_chat",
  "pad_token_id": 0,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "tokenizer_files": [
    "tokenizer.json",
    "tokenizer_config.json"
  ],
  "version": "0.1.0"
}

```

4. 模型 lib 编译

```
# 使用 mlc-chat-config.json 配置编译生成模型 lib 库（so 文件）
mlc_llm compile ./dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/mlc-chat-config.json \
    --device cuda \
    -o dist/libs/RedPajama-INCITE-Chat-3B-v1-q4f16_1-cuda.so

# compile with option
mlc_llm compile ./dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/mlc-chat-config.json \
    --device cuda -o  dist/libs/RedPajama-INCITE-Chat-3B-v1-q4f16_1-cuda.so \
    --opt             flashinfer=0;cublas_gemm=0;faster_transformer=1;cudagraph=0

# 生成的 so 文件，大小为 1.9MB
# ll RedPajama-INCITE-Chat-3B-v1-q4f16_1-cuda.so
-rwxr-xr-x 1 root root 1.9M Mar 12 14:56 RedPajama-INCITE-Chat-3B-v1-q4f16_1-cuda.so

```

### 模型执行流程

一旦模型权重、模型 lib 库和 Chat 配置准备好，MLC runtime 就可以将它们作为引擎来驱动聊天应用程序。下面的图表展示了 MLC 聊天应用程序的典型工作流程。

![](./assets/mlc_runtime.png)

所有 MLC runtime，包括 iOS、Web、CLI 等，都使用上面这三个元素。所有 runtime 都可以读取相同的模型权重文件。模型 lib 的打包方式可能因不同环境的 runtime  而异。

- Python 模型执行示例

```
from mlc_chat import ChatModule
from mlc_chat.callback import StreamToStdout

cm = ChatModule(
    model="/dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/", # Chat 配置和模型在同一个目录
    model_lib_path="/dist/libs/RedPajama-INCITE-Chat-3B-v1-q4f16_1-cuda.so"
    # Vulkan on Linux: Llama-2-7b-chat-hf-q4f16_1-vulkan.so
    # Metal on macOS: Llama-2-7b-chat-hf-q4f16_1-metal.so
    # Other platforms: Llama-2-7b-chat-hf-q4f16_1-{backend}.{suffix}
)
cm.generate(prompt="What is the meaning of life?", progress_callback=StreamToStdout(callback_interval=2))

```

输出示例：

```
# python test.py
[2024-03-14 02:18:24] INFO auto_device.py:76: Found device: cuda:0
[2024-03-14 02:18:25] INFO auto_device.py:85: Not found device: rocm:0
[2024-03-14 02:18:26] INFO auto_device.py:85: Not found device: metal:0
[2024-03-14 02:18:27] INFO auto_device.py:85: Not found device: vulkan:0
[2024-03-14 02:18:28] INFO auto_device.py:85: Not found device: opencl:0
[2024-03-14 02:18:28] INFO auto_device.py:33: Using device: cuda:0
[2024-03-14 02:18:28] INFO chat_module.py:373: Using model folder: /data/mlc-llm/compile_Llama-2-7b-chat/dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC
[2024-03-14 02:18:28] INFO chat_module.py:374: Using mlc chat config: /dist/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/mlc-chat-config.json
[2024-03-14 02:18:28] INFO chat_module.py:516: Using library model: dist/libs/RedPajama-INCITE-Chat-3B-v1-q4f16_1-cuda.so
[2024-03-14 02:18:29] INFO model_metadata.py:97: Total memory usage: 1581.73 MB (Parameters: 1491.34 MB. KVCache: 0.00 MB. Temporary buffer: 90.39 MB)
[2024-03-14 02:18:29] INFO model_metadata.py:106: To reduce memory usage, tweak `prefill_chunk_size`, `context_window_size` and `sliding_window_size`

The meaning of life is a concept that has been explored by many different people throughout history. Many people believe that the meaning of life is to find a purpose, to find out what it is that you were created for. Others believe that the meaning of life is to achieve happiness, to find meaning in your life. Still, others believe that the meaning of life is to find a sense of purpose, to find out what you are here for. No matter what your beliefs are, it is important to remember that the meaning of life is something that you have to find for yourself.

```

- RestAPI 示例

启动 REST 服务

```
# python -m mlc_llm.rest  --model dist/Llama-2-7b-chat-hf_q4f16_1-MLC_convert/ \
 --lib-path   dist/libs/Llama-2-7b-chat-hf-q4f16_1-cuda.so  --host 127.0.0.1 --port 8765
INFO:     Started server process [9049]
INFO:     Waiting for application startup.
[2024-03-14 02:37:55] INFO auto_device.py:76: Found device: cuda:0
[2024-03-14 02:37:56] INFO auto_device.py:85: Not found device: rocm:0
[2024-03-14 02:37:57] INFO auto_device.py:85: Not found device: metal:0
[2024-03-14 02:37:58] INFO auto_device.py:85: Not found device: vulkan:0
[2024-03-14 02:37:59] INFO auto_device.py:85: Not found device: opencl:0
[2024-03-14 02:37:59] INFO auto_device.py:33: Using device: cuda:0
[2024-03-14 02:37:59] INFO chat_module.py:373: Using model folder: /data/mlc-llm/dist/Llama-2-7b-chat-hf_q4f16_1-MLC_convert
[2024-03-14 02:37:59] INFO chat_module.py:374: Using mlc chat config: dist/Llama-2-7b-chat-hf_q4f16_1-MLC_convert/mlc-chat-config.json
[2024-03-14 02:37:59] INFO chat_module.py:516: Using library model: dist/libs/Llama-2-7b-chat-hf-q4f16_1-cuda.so
[2024-03-14 02:38:00] INFO model_metadata.py:96: Total memory usage: 4077.76 MB (Parameters: 3615.13 MB. KVCache: 0.00 MB. Temporary buffer: 462.62 MB)
[2024-03-14 02:38:00] INFO model_metadata.py:105: To reduce memory usage, tweak `prefill_chunk_size`, `context_window_size` and `sliding_window_size`
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8765 (Press CTRL+C to quit)
```

client 调用代码

```
import requests
import json

# Get a response using a prompt without streaming
payload = {
   "model": "vicuna-v1-7b",
   "messages": [{"role": "user", "content": "Write a haiku"}],
   "stream": False,
   "stop": "</s>"
}
r = requests.post("http://127.0.0.1:8765/v1/chat/completions", json=payload)
print(f"Without streaming:\n{r.json()['choices'][0]['message']['content']}\n")

```

```
# python client.py
Without streaming:
Sure! Here is a haiku about the sun:

Sun shines bright and warm
Bringing light and life to all
Nature's burning ball

Would you like me to write another haiku?

```
Other Prompt example：
```
system ="Your are an expert on C++ programing, help to answer user's question. "
user = "Please give me the C++ style code to return all the Fibonacci numbers under 100."
prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"
```

多进程调用
```
import requests
import json
from datetime import datetime


import multiprocessing

def worker(process_num, prompt):
   payload = {
   "model": "vicuna-v1-7b",
   "messages": [{"role": "user", "content": prompt}],
   "stream": False,
   "stop": "</s>"
    }
   totoal_time = 0
   for i in range(5):
       start_time = datetime.now()
       r = requests.post("http://127.0.0.1:8765/v1/chat/completions", json=payload)
       end_time = datetime.now()
       time_difference = end_time - start_time
       seconds = time_difference.total_seconds()
       print("process: {}, index: {}, time: {}".format(process_num, i, seconds))
       totoal_time = totoal_time + seconds
   print("process: {}, total time: {}".format(process_num, totoal_time))

pool = multiprocessing.Pool(processes=3)

for i in range(3):
    pool.apply_async(worker, args=(i, prompt))

pool.close()

pool.join()

```

### profilling

在 jetson 上用 nsys profilling 执行一次 Llama7B 的结果。

```

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)   Max (ns)   StdDev (ns)           Name
 --------  ---------------  ---------  -----------  ---------  --------  ----------  -----------  ----------------------
     63.0      13435232096       5931    2265255.8    15360.0      1088   101322400    8141699.0  ioctl
     15.5       3311622048         38   87147948.6   344208.0      3168  2933323008  475231760.5  poll
     10.8       2300231232          6  383371872.0   103584.0    101088  2299715680  938812900.3  waitpid
      5.4       1143954304      19688      58104.1     3488.0       832    22718560     717496.4  read
      2.5        536381536         85    6310371.0  4843808.0       256    34713952    6945485.8  pthread_mutex_lock
      1.6        341261376      39190       8707.9     8128.0      1664       64480       3674.6  stat64
      0.7        142887264      10196      14014.1    13248.0      7584       70048       3153.8  open64
      0.3         59996160      19517       3074.0     3072.0      1920       70624       1094.4  fstat64
      0.1         27615936     763667         36.2       32.0         0       15616         49.8  pthread_cond_signal
      0.1         11116192         83     133930.0    12480.0      4000     9804320    1074462.5  fopen
      0.0          7382912        393      18786.0    13376.0      3232      252256      30298.7  mmap
      0.0          5545696        149      37219.4    42240.0      5248       54368      13379.2  fopen64
      0.0          5153856        287      17957.7    14464.0      4896      590880      35829.1  mmap64
      0.0          3581248        906       3952.8     3200.0      2240       21056       1812.2  lstat64
      0.0          1967136        159      12371.9    13920.0      3072       17120       3597.6  fclose
      0.0          1852832         47      39422.0    10016.0      5024      294208      82882.8  open
      0.0          1365760        970       1408.0      832.0       544       23040       1266.7  fcntl
      0.0           986304          9     109589.3    35776.0     15136      680640     215264.9  sem_wait
      0.0           569920        885        644.0      480.0         0        5696        614.1  sigaction
      0.0           528896         59       8964.3     7392.0      2464       20416       4756.7  write
      0.0           352672          1     352672.0   352672.0    352672      352672          0.0  pthread_create
      0.0           256640         32       8020.0     8320.0      4032       11616       1472.9  stat
      0.0           243840          7      34834.3    34752.0     33632       36128        816.9  socket
      0.0           238720        220       1085.1       96.0        32       77760       7334.8  fgets
      0.0           221248         17      13014.6     9088.0      4480       23808       7356.0  pipe2
      0.0           129184          8      16148.0    17328.0      8128       18528       3319.5  fread
      0.0           114752          7      16393.1    16608.0     14848       17024        718.6  bind
      0.0            68768          8       8596.0     7680.0      6208       12160       2321.6  munmap
      0.0            52096        169        308.3      192.0         0        1856        386.3  fflush
      0.0            43392         21       2066.3     1856.0      1056        3232        761.8  dup
      0.0            25440          7       3634.3     3552.0      3328        3936        253.2  fstat
      0.0            20000         10       2000.0     1936.0      1280        3328        711.9  dup2
      0.0             7840         29        270.3      224.0        32         672        183.9  pthread_cond_broadcast
      0.0             4896        128         38.3       32.0         0         800         75.7  pthread_mutex_trylock
      0.0             3584          1       3584.0     3584.0      3584        3584          0.0  getc
      0.0             2752         57         48.3       32.0         0         640         84.5  flockfile

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)            Name
 --------  ---------------  ---------  ----------  ---------  --------  --------  -----------  -------------------------
     79.9      11829862720        735  16095051.3  2424128.0      4448  35817696   16864189.5  cudaFree
      9.6       1413790240        748   1890093.9    23232.0      5728  56953760    4635181.4  cudaMalloc
      6.1        909051392     108876      8349.4     7776.0      6240     85536       2558.4  cuLaunchKernel
      3.8        564974720       4680    120721.1     9568.0      4736   8364896     611331.6  cudaMemcpyAsync
      0.4         54165024       1665     32531.5     5472.0      2112    390688      83917.4  cudaStreamSynchronize
      0.1         17285888        748     23109.5    21808.0      5696    267616      12161.0  cudaMemGetInfo
      0.0          2396384        335      7153.4     6560.0      5792     24000       1553.6  cudaEventRecord
      0.0          2350976          1   2350976.0  2350976.0   2350976   2350976          0.0  cuModuleLoadData
      0.0          1975200        335      5896.1     4832.0      3680    147680       8070.3  cudaEventCreate
      0.0          1551264        335      4630.6     4352.0      3456      7776        892.2  cudaStreamWaitEvent
      0.0          1018496        335      3040.3     2944.0      2368      5472        531.6  cudaEventDestroy
      0.0           551072          1    551072.0   551072.0    551072    551072          0.0  cuModuleUnload
      0.0            64352          1     64352.0    64352.0     64352     64352          0.0  cudaStreamCreateWithFlags
      0.0            22912          1     22912.0    22912.0     22912     22912          0.0  cudaStreamDestroy
      0.0             5184          2      2592.0     2592.0      2464      2720        181.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                          Name
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -----------------------------------------------------
     39.1       5273427712      10688   493397.1   493472.0    491040    504096       1052.7  fused_fused_dequantize3_NT_matmul12_kernel
     22.2       2988560576      10688   279618.3   279648.0    276896    285632        645.9  fused_fused_dequantize1_NT_matmul10_kernel
     19.5       2622934560      10688   245409.3   245376.0    243488    251392        680.0  fused_fused_dequantize4_NT_matmul13_kernel
      7.7       1035412800      10688    96876.2    96896.0     94112     99776       1040.8  fused_fused_dequantize2_NT_matmul11_kernel
      5.4        732110848      10688    68498.4    68896.0     34304    105120      19922.9  batch_decode_paged_kv_kernel
      1.8        245820800        335   733793.4   733792.0    731456    745728       1344.8  fused_fused_dequantize_fused_NT_matmul14_cast1_kernel
      1.3        176675840      21440     8240.5     8096.0      7680     42656       1520.8  fuse_add_norm_prefill_kernel
      0.6         78797728      10720     7350.5     6912.0      5792    175072       8890.6  fused_rope_kernel
      0.5         66742144      10688     6244.6     6272.0      5184      7328        276.1  fused_split1_silu1_multiply1_kernel
      0.5         66470880      10720     6200.6     5952.0      5696     51008       2303.1  tir_kv_cache_transpose_append_kernel
      0.5         64960064         32  2030002.0  2029680.0   2022752   2038496       4438.5  fused_fused_dequantize3_NT_matmul7_kernel_2
      0.3         35686592         32  1115206.0  1112144.0   1104672   1132704       7984.9  fused_fused_dequantize1_NT_matmul5_kernel_2
      0.2         31208448         32   975264.0   974400.0    968960    986656       5239.0  fused_fused_dequantize4_NT_matmul8_kernel_2
      0.2         22450496        335    67016.4    66784.0     66144     83680       1075.0  softmax_kernel
      0.1         17897312         32   559291.0   558656.0    554816    576192       3740.3  batch_prefill_ragged_kv_kernel
      0.1         11972544         32   374142.0   373936.0    372576    377728       1158.6  fused_fused_dequantize2_NT_matmul6_kernel_2
      0.0          5539424         32   173107.0   173152.0    171392    173920        521.1  fused_split_silu_multiply_kernel
      0.0          3495008        335    10432.9    10368.0      9824     11296        371.1  divide_kernel
      0.0          2153184        335     6427.4     6048.0      5920     54464       2673.9  fused_fused_dequantize_take_kernel
      0.0          1770560        334     5301.1     5248.0      5120      7712        248.1  rms_norm1_kernel
      0.0            35360          1    35360.0    35360.0     35360     35360          0.0  rms_norm_kernel
      0.0             5632          1     5632.0     5632.0      5632      5632          0.0  index_kernel

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)           Operation
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ----------------------------
     99.5        453712160   4345  104421.7    1856.0       544   8161792     612465.5  [CUDA memcpy Host-to-Device]
      0.5          2176416    335    6496.8    6400.0      6368      8640        359.5  [CUDA memcpy Device-to-Host]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)           Operation
 ----------  -----  --------  --------  --------  --------  -----------  ----------------------------
   3790.785   4345     0.872     0.000     0.000    65.536        5.052  [CUDA memcpy Host-to-Device]
     42.880    335     0.128     0.128     0.128     0.128        0.000  [CUDA memcpy Device-to-Host]

```

在 3090 上用 nsys profilling 执行一次 Llama7B 的结果。

### 总结

至此基本总结如上。