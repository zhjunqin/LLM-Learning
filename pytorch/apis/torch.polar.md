# torch.polar


## torch.polar 文档

```
torch.polar(abs, angle, *, out=None) → Tensor
```

构造一个复数张量，其元素是与绝对值 abs 和角度 angle 对应的笛卡尔坐标。

$$ \text{out} = \text{abs} \cdot \cos(\text{agnle}) + \text{abs} \cdot \sin(\text{agnle}) \cdot j $$

示例

```
>>> import numpy as np
>>> abs = torch.tensor([1, 2], dtype=torch.float64)
>>> angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
>>> z = torch.polar(abs, angle)
>>> z
tensor([(0.0000+1.0000j), (-1.4142-1.4142j)], dtype=torch.complex128)
```

## 详细介绍

torch.polar 是 PyTorch 中的一个函数，用于将极坐标（幅度和相位）转换为复数形式。具体来说，它将给定的幅度（magnitude）和相位（angle）转换为对应的复数张量。

## 数学公式

$$ \text{output} = \text{abs} \cdot e^{i \cdot \text{angle}} $$

其中：
- $\text{abs}$ 是幅度（模）。
- $\text{angle}$ 是相位角（以弧度为单位）。
- $e^{i \cdot \text{angle}}$ 是欧拉公式，表示复数的相位部分。

示例

```
import torch

# 定义幅度和相位
abs = torch.tensor([1.0, 2.0, 3.0])
angle = torch.tensor([0.0, torch.pi / 2, torch.pi])

# 使用 torch.polar 转换为复数
complex_tensor = torch.polar(abs, angle)

print(complex_tensor)
```

```
tensor([ 1.0000+0.0000j,  0.0000+2.0000j, -3.0000+0.0000j])
```

- 第一个元素的幅度为 $1.0$，相位为 $0.0$，因此对应的复数为 $1.0+0.0j$。
- 第二个元素的幅度为 $2.0$，相位为 $\pi/2$，因此对应的复数为 $0.0+2.0j$。
- 第三个元素的幅度为 3.0，相位为 $\pi$，因此对应的复数为 $−3.0+0.0j$。

## 注意事项
- abs 和 angle 的形状必须相同，或者可以广播到相同的形状。
- abs 必须是非负的，否则会抛出错误。
- angle 的单位是弧度。

## 参考文献
- http://www.hbase.cn/archives/1589.html
- https://docs.pytorch.org/docs/stable/generated/torch.polar.html