# Normalization

## Batch Normalization

## Layer Normalization

https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

平均值和标准差是在最后的 `D` 个维度上计算的，其中 D 是 `normalized_shape` 的维度。
例如，如果 `normalized_shape` 是 (3, 5)（一个 2 维的形状），那么平均值和标准差是在输入的最后 2 个维度上计算的（即 `input.mean((-2, -1))`）。
标准差通过有偏估计计算，相当于 `torch.var(input, unbiased=False)`, 注意：std 有个参数 `unbiased`。
如果 `elementwise_affine` 为 True，则 `γ` 和 `β` 是 `normalized_shape` 的可学习仿射变换参数。

```
import torch
import numpy as np
 
x = np.array([[[1,2,-1,1], [3,4,-2,2]],
              [[1,2,1,1], [3,4,2,2]]], dtype=np.float32)
x_tensor = torch.from_numpy(x)
x_tensor


ln = torch.nn.LayerNorm(normalized_shape=4, eps=0, elementwise_affine=False)
ln(x_tensor)

x_mean = x_tensor.mean(-1, keepdim=True)
x_mean
x_std = x_tensor.std(-1, keepdim=True, unbiased=False)
x_std
(x_tensor - x_mean) / x_std
```

```
>>> ln(x_tensor)
tensor([[[ 0.2294,  1.1471, -1.6059,  0.2294],
         [ 0.5488,  0.9879, -1.6465,  0.1098]],

        [[-0.5774,  1.7321, -0.5774, -0.5774],
         [ 0.3015,  1.5076, -0.9045, -0.9045]]])
>>> x_mean = x_tensor.mean(-1, keepdim=True)
>>> x_mean
tensor([[[0.7500],
         [1.7500]],

        [[1.2500],
         [2.7500]]])
>>> x_std = x_tensor.std(-1, keepdim=True, unbiased=False)
>>> x_std
tensor([[[1.0897],
         [2.2776]],

        [[0.4330],
         [0.8292]]])
>>> (x_tensor - x_mean) / x_std
tensor([[[ 0.2294,  1.1471, -1.6059,  0.2294],
         [ 0.5488,  0.9879, -1.6465,  0.1098]],

        [[-0.5774,  1.7321, -0.5774, -0.5774],
         [ 0.3015,  1.5076, -0.9045, -0.9045]]])
```