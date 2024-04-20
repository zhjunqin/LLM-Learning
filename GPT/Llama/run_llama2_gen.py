
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.profiler import profile, ProfilerActivity

model_dir = "./Llama-2-7b-chat-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_dir, device_map="auto", )

# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
model = LlamaForCausalLM.from_pretrained(
    model_dir, device_map="auto", torch_dtype=torch.float16)
model = model.eval()

input_text = "<s>[INST] <<SYS>>\nYour are an expert on C++ programing, help to answer user's question. \n<</SYS>>\n\nPlease give me the C++ style code to return all the Fibonacci numbers under 100. [/INST]"
inputs = tokenizer(input_text, return_tensors="pt").to('cuda')

# Generate
generate_ids = model.generate(
    inputs.input_ids, max_length=1024,
    do_sample=True,
    temperature=0.6,
    top_p=0.7,  # cause error
    repetition_penalty=(1.0 / 0.85),
)
result = tokenizer.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(result)


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace(
        "./profiling/llama_gen_trace_{}.json".format(str(p.step_num)))


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
    inputs = tokenizer(input_text, return_tensors="pt").to('cuda')

    # Generate
    generate_ids = model.generate(
        inputs.input_ids, max_length=1024,
        do_sample=True,
        temperature=0.6,
        top_p=0.7,  # cause error
        repetition_penalty=(1.0 / 0.85),
    )
    result = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    print(result)
    p.step()

"""
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                               aten::mm        12.11%        1.482s        16.84%        2.062s      27.357us        5.818s        85.66%        5.989s      79.455us           0 b           0 b       1.03 Gb       1.03 Gb         75375
ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f...         0.00%       0.000us         0.00%       0.000us       0.000us        3.126s        46.03%        3.126s      58.503us           0 b           0 b           0 b           0 b         53440
ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64...         0.00%       0.000us         0.00%       0.000us       0.000us        2.555s        37.62%        2.555s     119.540us           0 b           0 b           0 b           0 b         21376
                                       cudaLaunchKernel        20.40%        2.498s        20.40%        2.498s       5.227us     546.254ms         8.04%     546.267ms       1.143us           0 b           0 b     -84.50 Kb     -84.50 Kb        477988
                                              aten::cat         4.54%     555.833ms         6.37%     779.489ms      17.925us     197.092ms         2.90%     246.714ms       5.673us           0 b           0 b      38.44 Gb      38.44 Gb         43486
                     aten::_efficient_attention_forward         1.62%     198.859ms         3.49%     427.127ms      39.844us     191.814ms         2.82%     229.316ms      21.391us     167.50 Kb      31.36 Kb      99.50 Mb           0 b         10720
fmha_cutlassF_f16_aligned_64x128_rf_sm80(PyTorchMemE...         0.00%       0.000us         0.00%       0.000us       0.000us     191.814ms         2.82%     191.814ms      17.893us           0 b           0 b           0 b           0 b         10720
                                              aten::mul         8.66%        1.061s        12.42%        1.521s      15.441us     151.408ms         2.23%     265.772ms       2.698us          -8 b          -8 b       1.24 Gb       1.24 Gb         98490
void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us     131.527ms         1.94%     131.527ms       6.171us           0 b           0 b           0 b           0 b         21312
void cutlass::Kernel<cutlass_80_tensorop_f16_s16816g...         0.00%       0.000us         0.00%       0.000us       0.000us     123.635ms         1.82%     123.635ms     310.641us           0 b           0 b           0 b           0 b           398
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 12.245s
Self CUDA time total: 6.792s
"""
