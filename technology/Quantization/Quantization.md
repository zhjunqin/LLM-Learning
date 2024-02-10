# 权重量化简介

大型语言模型（LLM）以其庞大的计算需求而闻名。通常，模型的大小是通过参数数量（大小）乘以这些值的精度（数据类型）来计算。然而，为了节省内存，可以通过量化使用低精度的数据类型存储权重。

两个主要的权重量化技术类别：

- Post-Training Quantization（PTQ）是一种简单直接的技术，将已经训练好的模型的权重转换为低精度，而无需重新训练。尽管实施简单，但 PTQ 可能会导致性能下降。

- Quantization-Aware Training（QAT）在预训练或微调阶段中加入了权重转换过程，从而改善了模型的性能。然而，QAT 在计算上是昂贵的，并且需要具有代表性的训练数据。

这里重点介绍 PTQ，以降低参数的精度。为了更好地理解，我们将使用一个 GPT-2 模型的示例，应用简单和更复杂的技术。

## 精度介绍

详细参考[精度介绍一文](../precision/fp32_fp16_bf16.md)

## 朴素的 8-bit 量化

在本节中，我们将实施两种量化技术：一种是 absolute maximum（absmax）量化的对称量化方法，另一种是 zero-point 量化的非对称量化方法。在这两种情况下，目标是将 FP32 张量 $X$ (原始权重）映射到 INT8 张量 $X_{quant}$（量化权重）。

### absmax quantization
使用 absmax 量化，原始数值被除以张量的绝对最大值，并乘以一个缩放因子（127），将输入映射到范围 [-127, 127]。为了恢复原始的 FP16 值，INT8 数值除以量化因子，但是四舍五入会导致一定的精度损失。

$$
X_{\rm quant} = \rm round \left( \frac{127}{\rm max{|X|}} \cdot X \right) 
$$
$$
X_{\rm dequant} = \frac{\rm max{|X|}}{127} \cdot X_{\rm quant}
$$

例如，假定有一个绝对最大值为 3.2。权重为 0.1 将被量化为 $\rm round \left( \frac{127}{\rm 3.2} \times 0.1 \right) = \rm round (3.96875) = 4 $。

如果想要将其反量化，我们将得到 $\frac{3.2}{127} \times 4 = 0.1008 $，这意味着一个误差为 0.008。

下面是相应的 Python 实现：

```
import torch

def absmax_quantize(X):
    # Calculate scale
    scale = 127 / torch.max(torch.abs(X))

    # Quantize
    X_quant = (scale * X).round()

    # Dequantize
    X_dequant = X_quant / scale

    return X_quant.to(torch.int8), X_dequant
```

### zero-point quantization

使用 zero-point 量化，我们可以考虑非对称的输入分布，在考虑 ReLU 函数的输出（仅为正值）时很有用。首先，输入值将通过值的总范围（255）除以最大值和最小值之间的差异来进行缩放。然后，此分布通过 zero-point 进行偏移，将其映射到范围 [-128, 127]（与 absmax 相比增加了 zero-point）。首先，我们计算缩放因子( scale factor )和零点值( zero-point )：

$$
\rm scale =  \frac{255}{\rm max(X) - \rm min(X)}
$$

$$
\rm zeropoint = - \rm round(scale \cdot min(X)) - 128
$$

然后，我们可以用这些变量来进行量化和反量化。

$$
X_{\rm quant} = \rm round \left( \rm scale \cdot X + zeropoint \right) 
$$

$$
X_{\rm dequant} = \frac{X_{\rm quant} - \rm zeropoint}{\rm scale} 
$$

假定有一个最大值为 3.2 和最小值为 -3.0，可以计算 $\rm scale = \frac{255}{ 3.2 + 3.0}  = 41.13$ 和 $\rm zeropoint = - \rm round(41.13 \cdot -3.0) - 128 = 123 - 128 = -5$，则之前的 0.1 量化为 $\rm round \left( \rm 41.13 \cdot 0.1 -5 \right) = -1$，得到的结果和之前的 absmax 不同。

![](./assets/naive_quantization.png)

Python 的实现如下：

```
def zeropoint_quantize(X):
    # Calculate value range (denominator)
    x_range = torch.max(X) - torch.min(X)
    x_range = 1 if x_range == 0 else x_range

    # Calculate scale
    scale = 255 / x_range

    # Shift by zero-point
    zeropoint = (-scale * torch.min(X) - 128).round()

    # Scale and round the inputs
    X_quant = torch.clip((X * scale + zeropoint).round(), -128, 127)

    # Dequantize
    X_dequant = (X_quant - zeropoint) / scale

    return X_quant.to(torch.int8), X_dequant
```

## 使用 transformers 库

我们可以借助 transformers 库在真实模型上使用这两个函数。

```
pip install -q bitsandbytes>=0.39.0
pip install -q accelerate
pip install transformers
```

我们首先加载 GPT-2 的模型和分词器。这是一个非常小的模型，我们可能不想对其进行量化，但对于本教程来说已经足够了。首先，我们想观察模型的大小，以便稍后进行比较，并评估由于 8 位量化而产生的内存节省。

```
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(0)

# Set device to CPU for now
device = 'cpu'

# Load model and tokenizer
model_id = 'gpt2'
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Print model size
print(f"Model size: {model.get_memory_footprint():,} bytes")
```
得到模型参数大小
```
Model size: 510,342,192 bytes
```
GPT-2 模型参数 FP32 精度的大小大约为 487MB。

下一步是使用 zero-point 量化和 absmax 量化对权重进行量化。在下面的示例中，我们将这些技术应用于 GPT-2 的第一个注意力层，以查看结果。

```
# Extract weights of the first layer
weights = model.transformer.h[0].attn.c_attn.weight.data
print("Original weights:")
print(weights)

# Quantize layer using absmax quantization
weights_abs_quant, _ = absmax_quantize(weights)
print("\nAbsmax quantized weights:")
print(weights_abs_quant)

# Quantize layer using absmax quantization
weights_zp_quant, _ = zeropoint_quantize(weights)
print("\nZero-point quantized weights:")
print(weights_zp_quant)
```

```
Original weights:
tensor([[-0.4738, -0.2614, -0.0978,  ...,  0.0513, -0.0584,  0.0250],
        [ 0.0874,  0.1473,  0.2387,  ..., -0.0525, -0.0113, -0.0156],
        [ 0.0039,  0.0695,  0.3668,  ...,  0.1143,  0.0363, -0.0318],
        ...,
        [-0.2592, -0.0164,  0.1991,  ...,  0.0095, -0.0516,  0.0319],
        [ 0.1517,  0.2170,  0.1043,  ...,  0.0293, -0.0429, -0.0475],
        [-0.4100, -0.1924, -0.2400,  ..., -0.0046,  0.0070,  0.0198]])

Absmax quantized weights:
tensor([[-21, -12,  -4,  ...,   2,  -3,   1],
        [  4,   7,  11,  ...,  -2,  -1,  -1],
        [  0,   3,  16,  ...,   5,   2,  -1],
        ...,
        [-12,  -1,   9,  ...,   0,  -2,   1],
        [  7,  10,   5,  ...,   1,  -2,  -2],
        [-18,  -9, -11,  ...,   0,   0,   1]], dtype=torch.int8)

Zero-point quantized weights:
tensor([[-20, -11,  -3,  ...,   3,  -2,   2],
        [  5,   8,  12,  ...,  -1,   0,   0],
        [  1,   4,  18,  ...,   6,   3,   0],
        ...,
        [-11,   0,  10,  ...,   1,  -1,   2],
        [  8,  11,   6,  ...,   2,  -1,  -1],
        [-18,  -8, -10,  ...,   1,   1,   2]], dtype=torch.int8)

```
原始（FP32）和量化（INT8）值之间的差异比较明显，但是 absmax 权重和 zero-point 权重之间的差异比较微妙。在这种情况下，看起来 zero-point 被用 -1 进行了偏移。这表明该层的权重分布相当对称。




## 参考
- https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html