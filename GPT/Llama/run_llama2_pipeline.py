import torch
import transformers

from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.profiler import profile, record_function, ProfilerActivity

model_dir = "./Llama-2-7b-chat-hf"
model = LlamaForCausalLM.from_pretrained(
    model_dir, device_map="cuda", torch_dtype=torch.float16)

tokenizer = LlamaTokenizer.from_pretrained(model_dir, device_map="cuda")

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline('I have tomatoes, basil and cheese at home. What can I cook for dinner?\n',
                     do_sample=True,
                     top_k=10,
                     num_return_sequences=1,
                     eos_token_id=tokenizer.eos_token_id,
                     max_length=400,
                     )

for seq in sequences:
    print(f"{seq['generated_text']}")


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace(
        "./profiling/llama_trace_{}.json".format(str(p.step_num)))


with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
    # schedule=torch.profiler.schedule(
    #     wait=1,
    #     warmup=1,
    #     active=3),
    on_trace_ready=trace_handler
) as p:
    sequences = pipeline('I have tomatoes, basil and cheese at home. What can I cook for dinner?\n',
                         do_sample=True,
                         top_p=0.6,
                         num_return_sequences=1,
                         eos_token_id=tokenizer.eos_token_id,
                         max_length=400,
                         )
    for seq in sequences:
        print(f"{seq['generated_text']}")
    p.step()


"""
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                               aten::mm        11.69%     799.358ms        16.48%        1.128s      28.312us        3.069s        87.46%        3.159s      79.329us           0 b           0 b     522.99 Mb     522.99 Mb         39825
ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f...         0.00%       0.000us         0.00%       0.000us       0.000us        1.643s        46.84%        1.643s      58.356us           0 b           0 b           0 b           0 b         28160
ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64...         0.00%       0.000us         0.00%       0.000us       0.000us        1.345s        38.34%        1.345s     119.425us           0 b           0 b           0 b           0 b         11264
                                       cudaLaunchKernel        21.66%        1.481s        21.66%        1.481s       5.896us     278.461ms         7.94%     278.465ms       1.108us           0 b           0 b     -69.00 Kb     -69.00 Kb        251280
                                              aten::mul         8.26%     564.813ms        12.31%     841.762ms      16.231us      77.491ms         2.21%     135.857ms       2.620us           0 b           0 b     629.74 Mb     629.74 Mb         51861
                                              aten::cat         4.60%     314.581ms         6.67%     455.989ms      19.872us      75.155ms         2.14%     100.942ms       4.399us           0 b           0 b       9.69 Gb       9.69 Gb         22946
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us      69.409ms         1.98%      69.409ms     289.204us           0 b           0 b           0 b           0 b           240
                     aten::_efficient_attention_forward         1.63%     111.776ms         3.47%     237.069ms      41.855us      61.567ms         1.75%      81.042ms      14.308us      88.50 Kb      15.84 Kb      49.25 Mb           0 b          5664
fmha_cutlassF_f16_aligned_64x128_rf_sm80(PyTorchMemE...         0.00%       0.000us         0.00%       0.000us       0.000us      61.567ms         1.75%      61.567ms      10.870us           0 b           0 b           0 b           0 b          5664
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      48.112ms         1.37%      48.112ms       2.118us           0 b           0 b           0 b           0 b         22721
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 6.840s
Self CUDA time total: 3.508s
"""
