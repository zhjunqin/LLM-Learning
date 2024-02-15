# Llama2 fine tune

本文介绍为什么指令调整有效，并学习如何在创建自己的 Llama2 模型。

## LLM 微调介绍

LLM（Large Language Models）是在大规模文本语料库上进行预训练的。就 Llama2 而言，除了其 2 万亿 token 的长度外，我们对训练集的组成了解甚少。相比之下，BERT（2018）仅在 BookCorpus（8 亿个单词）和英文维基百科（ 25 亿个单词）上进行了训练。根据经验，这是一个非常昂贵且耗时的过程，存在许多硬件问题。如果您想了解更多信息，我建议阅读 Meta 关于 [OPT-175B](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf) 模型的预训练的日志。

当预训练完成后，像 Llama2 这样的自回归模型可以预测序列中的下一个 token。然而，这并不意味着它们特别有用，因为它们无法回答指令。这就是为什么我们采用指令微调来使它们的回答与人类的期望相一致。有两种主要的微调技术：

- Supervised Fine-Tuning（SFT）：模型在一组指令和回答的数据集上进行训练。它调整 LLM 中的权重，以最小化生成的答案与真实回答标签之间的差异。

- Reinforcement Learning from Human Feedback（RLHF）：模型通过与环境交互并接收反馈来进行学习。它们被训练以最大化奖励信号（使用 [PPO](https://arxiv.org/abs/1707.06347)），这个奖励信号通常源自人类对模型输出的评估。

一般来说，强化学习从人类反馈中学习到更复杂和微妙的人类偏好，但也更具挑战性，实现起来更为困难。事实上，它需要对奖励系统进行精心设计，并且对人类反馈的质量和一致性非常敏感。未来的一个可能的替代方法是直接偏好优化（[Direct Preference Optimization](https://arxiv.org/abs/2305.18290) DPO）算法，它在 SFT 模型上直接进行偏好学习。

在本文，我们将进行 SFT，但这引发了一个问题：为什么微调一开始就有效呢？正如 [Orca 论文](https://mlabonne.github.io/blog/notes/Large%20Language%20Models/orca.html)中强调的那样，我们的理解是微调利用了预训练过程中学到的知识。换句话说，如果模型从未见过您感兴趣的数据类型，微调几乎没有帮助。然而，预训练学习过这些知识，SFT 可以表现出极高的性能。

例如，[LIMA 论文](https://mlabonne.github.io/blog/notes/Large%20Language%20Models/lima.html)展示了如何通过对只有 1,000 个高质量样本进行 65 亿参数的 LLaMA（v1）模型进行微调，超越 GPT-3（DaVinci003）。达到这种性能水平的关键是指令数据集的质量，因此很多工作都集中在解决这个问题上（如 evol-instruct、Orca 或 phi-1）。请注意，LLM 的大小（65b，而不是 13b 或 7b）对于有效利用预先存在的知识也是至关重要的。

与数据质量相关的另一个重要点是提示词模板(prompt template)。提示词由类似的元素组成：系统提示词（可选）用于指导模型，用户提示词（必需）用于给出指令，附加输入（可选）用于考虑，以及模型的回答（必需）。就 Llama2 而言，使用了以下的聊天模型模板：

```
<s>[INST] <<SYS>>
System prompt
<</SYS>>

User prompt [/INST] Model answer </s>
```

还有其他模板，比如 Alpaca 和 Vicuna 的模板，这些模板的影响并不是很清楚。在本例中，我们将重新格式化我们的指令数据集，以符合 Llama2 的模板。本文使用了优秀的 [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) 数据集完成了这一步骤。您可以在 Hugging Face 上找到它，名称为 [mlabonne/guanaco-llama2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k)。接下来，我们将使用基础模型而不是聊天模型，所以这一步骤是可选的。请注意，如果您使用的是基础的 Llama2 模型而不是聊天版本，则不需要遵循特定的提示模板。

## 如何微调 Llama2

在本节中，我们将对具有 7B 参数的 Llama2 模型进行微调。Llama2-7b 的权重有 7B × 2byte = 14GB（使用 FP16 编码）。微调训练过程中的优化器状态、梯度和前向激活会产生的额外的显存开销，这让我们在具备 24G 或者更低显存的 GPU 无法训练。因此我们需要使用参数高效的微调（parameter-efficient fine-tuning PEFT）技术，如 LoRA 或 QLoRA。

为了大幅减少 VRAM 的使用量，我们必须以 4-bit 精度进行微调，因此我们将在这里使用 QLoRA。好消息是，我们可以利用 Hugging Face 生态系统中的 transformers、accelerate、peft、trl 和 bitsandbytes 库。以下是基于 Younes Belkada 的 [GitHub Gist](https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da) 的代码，我们将在其中进行操作。首先，我们安装并加载这些库。


```
!pip install accelerate peft bitsandbytes transformers trl tensorboard
```


```
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
```


然后我们执行一下加载的模型 llama-2-7b-chat-hf，尝试问几个问题，看会得到什么回复，后面跟微调过的模型进行对比。

```
model_name = "NousResearch/llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": 0},
    trust_remote_code=True,
    torch_dtype=torch.bfloat16).eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

prompt = "What is a large language model?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
```

```

```


让我们稍微谈谈我们可以在这里调整的参数。首先，我们想要加载一个 llama-2-7b-chat-hf 模型，并在 mlabonne/guanaco-llama2-1k（1,000 个样本）上进行训练，这将生成我们的微调模型 llama-2-7b-miniguanaco。如果您对如何创建这个数据集感兴趣，可以查看这个notebook。随意更改它：Hugging Face Hub上有许多很好的数据集，如databricks/databricks-dolly-15k。

QLoRA 将使用 64 rank 和 16 lora_alpha。我们将使用 NF4 类型直接以 4-bit 精度加载 Llama2 模型，并对其进行 1 个 epoch 的训练。要获取有关其他参数的更多信息，请查阅 TrainingArguments、PeftModel 和 SFTTrainer 的文档。

```
# The model that you want to train from the Hugging Face hub
model_name = "NousResearch/llama-2-7b-chat-hf"

# The instruction dataset to use
dataset_name = "mlabonne/guanaco-llama2-1k"

# Fine-tuned model name
new_model = "llama-2-7b-miniguanaco"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = "constant"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 25

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}
```

生成数据集 `mlabonne/guanaco-llama2-1k` 的代码：

```
from datasets import load_dataset
import re

# Load the dataset
dataset = load_dataset('timdettmers/openassistant-guanaco')

# Shuffle the dataset and slice it
dataset = dataset['train'].shuffle(seed=42).select(range(1000))

# Define a function to transform the data
def transform_conversation(example):
    conversation_text = example['text']
    segments = conversation_text.split('###')

    reformatted_segments = []

    # Iterate over pairs of segments
    for i in range(1, len(segments) - 1, 2):
        human_text = segments[i].strip().replace('Human:', '').strip()

        # Check if there is a corresponding assistant segment before processing
        if i + 1 < len(segments):
            assistant_text = segments[i+1].strip().replace('Assistant:', '').strip()

            # Apply the new template
            reformatted_segments.append(f'<s>[INST] {human_text} [/INST] {assistant_text} </s>')
        else:
            # Handle the case where there is no corresponding assistant segment
            reformatted_segments.append(f'<s>[INST] {human_text} [/INST] </s>')

    return {'text': ''.join(reformatted_segments)}


# Apply the transformation
transformed_dataset = dataset.map(transform_conversation)

transformed_dataset.push_to_hub("guanaco-llama2-1k")
```

我们现在可以加载所有内容并开始微调过程。

- 首先，我们要加载我们定义的数据集。在这里，我们的数据集已经经过预处理，但通常在这里您将重新格式化提示，过滤掉不良文本，合并多个数据集等。
- 然后，我们配置 bitsandbytes 以进行 4-bit 量化。
- 接下来，我们在相应的 tokenizer 上以 4-bit 精度加载 Llama2 模型。
- 最后，我们加载 QLoRA 的配置、常规训练参数，并将所有内容传递给 SFTTrainer。训练现在可以开始了！

```
# Load dataset (you can process it here)
dataset = load_dataset(dataset_name, split="train")

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)
```

我们可以通过以下方法在 TensorBoard 上查看图表：



为了确保模型的行为是正确的通常需要更详尽的评估，但我们可以使用文本生成流水线来提问诸如 “What is a large language model?” 的问题。请注意，这里需要格式化输入以匹配 Llama2 的提示词模板。

```
# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Run text generation pipeline with our next model
prompt = "What is a large language model?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
```

```
<s>[INST] What is a large language model? [/INST] A large language model is a type of artificial intelligence that is trained on a large dataset of text to generate human-like language. It is typically trained on a dataset of text that is much larger than the dataset used for smaller language models. The large dataset allows the model to learn more complex patterns in language, which can result in more accurate and natural-sounding language generation.

Large language models are often used for tasks such as text summarization, language translation, and chatbots. They are also used for more complex tasks such as writing articles, generating creative content, and even composing music.

Large language models are trained using a technique called deep learning, which involves using many layers of artificial neural networks to learn complex patterns in the data. The model is trained on a large dataset of text, and the neural networks are trained to predict the next word in a sequence of text given
```

模型输出如下的应答：

```
A large language model is a type of artificial intelligence that is trained on a large dataset of text to generate human-like language. It is typically trained on a dataset of text that is much larger than the dataset used for smaller language models. The large dataset allows the model to learn more complex patterns in language, which can result in more accurate and natural-sounding language generation.

Large language models are often used for tasks such as text summarization, language translation, and chatbots. They are also used for more complex tasks such as writing articles, generating creative content, and even composing music.

Large language models are trained using a technique called deep learning, which involves using many layers of artificial neural networks to learn complex patterns in the data. The model is trained on a large dataset of text, and the neural networks are trained to predict the next word in a sequence of text given
```

根据经验，对于只有 7B 参数的模型来说，它的连贯性非常好。您可以进行尝试，并从 [BigBench-Hard](https://github.com/suzgunmirac/BIG-Bench-Hard) 等评估数据集中问更难的问题。Guanaco 是一个产生过高质量模型的优秀数据集。您可以使用 [mlabonne/guanaco-llama2](https://huggingface.co/datasets/mlabonne/guanaco-llama2) 在整个数据集上训练一个 Llama2 模型。

现在我们如何存储我们的新模型 llama-2-7b-miniguanaco 呢？我们需要将 LoRA 的权重与基础模型合并。不幸的是，据我所知，没有直接的方法可以做到这一点：我们需要以 FP16 精度重新加载基础模型，并使用 peft 库来合并所有内容。

```
# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

我们已经合并了权重并重新加载了 tokenizer。现在，我们可以将所有内容推送到 Hugging Face Hub 以保存我们的模型。

```
!huggingface-cli login

model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)
```

您现在可以像从 Hub 加载任何其他 Llama2 模型一样使用该模型进行推理。您也可以重新加载它并使用另一个数据集来进一步微调。

建议使用一个单独的脚本来执行微调，在[这里](https://gist.github.com/mlabonne/8eb9ad60c6340cb48a17385c68e3b1a5)可以找到类似的代码。


## 总结

在本文中，我们介绍了如何对 Llama2-7b 模型进行微调。介绍了一些关于 LLM 训练和微调的必要背景知识，以及与指令数据集相关的重要考虑因素。在第二部分中，我们成功地使用原生提示词模板和自定义参数对 Llama2 模型进行了微调。


## 参考
- https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html
- https://www.datacamp.com/tutorial/fine-tuning-llama-2