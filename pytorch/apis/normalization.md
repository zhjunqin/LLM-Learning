# Normalization

## Batch Normalization

## Layer Normalization

https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

$$ y= \frac{x −E[x]}{ \sqrt {Var[x]+ϵ}}  ∗γ+β$$


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
tensor([[[0.7500],  # (1 + 2 + -1 + 1)/4
         [1.7500]], # (3 + 4 + -2 + 2)/4

        [[1.2500],  # (1 + 2 + 1 + 1)/4
         [2.7500]]])# (3 + 4 + 2 + 2)/4
>>> x_std = x_tensor.std(-1, keepdim=True, unbiased=False)
>>> x_std
tensor([[[1.0897],  # sqrt( ( (1-0.75)^2 + (2-0.75)^2 + (-1-0.75)^2 + (1-0.75)^2 ) /4 ) 
         [2.2776]],

        [[0.4330],
         [0.8292]]])
>>> (x_tensor - x_mean) / x_std
tensor([[[ 0.2294,  1.1471, -1.6059,  0.2294],
         [ 0.5488,  0.9879, -1.6465,  0.1098]],

        [[-0.5774,  1.7321, -0.5774, -0.5774],
         [ 0.3015,  1.5076, -0.9045, -0.9045]]])
```


## F.layer_norm 和 nn.LayerNorm


https://blog.csdn.net/weixin_45019478/article/details/115027728

### functional.layer_norm

```
torch.nn.functional.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05)
```

https://pytorch.org/docs/stable/generated/torch.nn.functional.layer_norm.html

这两者的区别在于 `F.layer_norm` 在外部定义可学习参数 $γ$ 和 $β$, 并在调用时传入，这两者的 shape 相同。

而 `nn.LayerNorm` 是通过 `elementwise_affine` 参数控制的。

示例：

```
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```