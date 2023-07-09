# TorchScript 

TorchScript 是 Pytorch 模型的中间表达，能够运行在高性能的环境，比如 C++ 的环境中。

## Tracing

TorchScript 提供了捕获模型定义的工具，即使在 PyTorch 的灵活和动态特性下也是如此。让我们首先来看看我们所说的跟踪 (tracing)。

```
import torch

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

my_cell = MyCell()
x, h = torch.rand(3, 4), torch.rand(3, 4)
traced_cell = torch.jit.trace(my_cell, (x, h))
print("=== Tracing ===")
print(traced_cell)
print("=== Tracing call ===")
print(traced_cell(x, h))
print("=== Tracing code ===")
print(traced_cell.code)
print("=== Tracing graph ===")
```

`torch.jit.trace` 调用了模块，在运行模块时记录了操作，并创建了 `torch.jit.ScriptModule` 的一个实例（TracedModule 的一个实例）。

TorchScript 在中间表示（IR）中记录其定义，通常在深度学习中称为 `graph`。我们可以使用 `.graph` 属性检查这个 `graph`。

`graph` 是一个非常底层的表示形式，包含的大部分信息对最终用户并不有用。相反，我们可以使用 `.code` 属性来给出代码的 Python 语法解释。

输出如下：

```
=== Tracing ===
<class 'torch.jit._trace.TopLevelTracedModule'>
MyCell(
  original_name=MyCell
  (linear): Linear(original_name=Linear)
)
=== Tracing call ===
(tensor([[ 0.8078,  0.3965, -0.2237,  0.5210],
        [ 0.6914, -0.4180,  0.1072,  0.7222],
        [ 0.8706, -0.1659, -0.6110,  0.7282]], grad_fn=<TanhBackward0>), tensor([[ 0.8078,  0.3965, -0.2237,  0.5210],
        [ 0.6914, -0.4180,  0.1072,  0.7222],
        [ 0.8706, -0.1659, -0.6110,  0.7282]], grad_fn=<TanhBackward0>))
=== Tracing code ===
def forward(self,
    x: Tensor,
    h: Tensor) -> Tuple[Tensor, Tensor]:
  linear = self.linear
  _0 = torch.tanh(torch.add((linear).forward(x, ), h))
  return (_0, _0)

=== Tracing graph ===
graph(%self.1 : __torch__.MyCell,
      %x : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu),
      %h : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu)):
  %linear : __torch__.torch.nn.modules.linear.Linear = prim::GetAttr[name="linear"](%self.1)
  %20 : Tensor = prim::CallMethod[name="forward"](%linear, %x)
  %11 : int = prim::Constant[value=1]() # /path/to/test.py:9:0
  %12 : Float(3, 4, strides=[4, 1], requires_grad=1, device=cpu) = aten::add(%20, %h, %11) # /path/to/test.py:9:0
  %13 : Float(3, 4, strides=[4, 1], requires_grad=1, device=cpu) = aten::tanh(%12) # /path/to/test.py:9:0
  %14 : (Float(3, 4, strides=[4, 1], requires_grad=1, device=cpu), Float(3, 4, strides=[4, 1], requires_grad=1, device=cpu)) = prim::TupleConstruct(%13, %13)
  return (%14)
```

那么这么做这一切的原因是什么呢？有几个原因：
- TorchScript 代码可以在自己的解释器中调用，这基本上是一个受限制的 Python 解释器。这个解释器不会获取全局解释器锁，因此许多请求可以在同一实例上同时处理。
- 这种格式允许我们将整个模型保存到磁盘并加载到另一个环境中，例如在使用 Python 以外的语言编写的服务器中。
- TorchScript 为我们提供了一种表示形式，在这种表示形式中，我们可以对代码进行编译器优化，以提供更高效的执行。
- TorchScript 允许我们与许多后端/设备运行时进行交互，这些运行时需要比单个操作符更广泛的程序视图。

调用 traced_cell 产生的结果与Python模块相同。

```
print(my_cell(x, h))
print(traced_cell(x, h))
```
```
(tensor([[-0.0060, -0.3173, -0.1889, -0.4302],
        [ 0.1956,  0.3746,  0.0606, -0.1401],
        [ 0.7122, -0.5074,  0.6233,  0.4109]], grad_fn=<TanhBackward0>), tensor([[-0.0060, -0.3173, -0.1889, -0.4302],
        [ 0.1956,  0.3746,  0.0606, -0.1401],
        [ 0.7122, -0.5074,  0.6233,  0.4109]], grad_fn=<TanhBackward0>))
(tensor([[-0.0060, -0.3173, -0.1889, -0.4302],
        [ 0.1956,  0.3746,  0.0606, -0.1401],
        [ 0.7122, -0.5074,  0.6233,  0.4109]],
       grad_fn=<DifferentiableGraphBackward>), tensor([[-0.0060, -0.3173, -0.1889, -0.4302],
        [ 0.1956,  0.3746,  0.0606, -0.1401],
        [ 0.7122, -0.5074,  0.6233,  0.4109]],
       grad_fn=<DifferentiableGraphBackward>))
```

## 参考文献

- https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html 
- https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html 
- https://pytorch.org/docs/stable/jit.html 
- https://pytorch.org/docs/master/jit_language_reference.html#language-reference