# TorchScript 

TorchScript 是 Pytorch 模型的中间表达，能够运行在高性能的环境，比如 C++ 的环境中。

## Tracing

TorchScript 提供了捕获模型定义的工具，即使在 PyTorch 的灵活和动态特性下也是如此。首先来看看跟踪 (tracing) 的功能。

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

### graph 的含义

虽然 graph 的输出是一个比较底层的代码，还是稍微看看里面内容的含义。

里面用到了底层的库 ATen。ATen 来自于 A TENsor library for C++11 的缩写，是一个张量库，几乎所有 PyTorch 中的其他 Python 和 C++ 接口都是在其之上构建的。它提供了一个核心 Tensor 类，上面定义了许多数百个操作。这些操作中的大多数都有 CPU 和 GPU 实现，Tensor 类会根据其类型动态分派到相应的实现。使用 ATen 的一个小例子如下所示：

```
#include <ATen/ATen.h>

at::Tensor a = at::ones({2, 2}, at::kInt);
at::Tensor b = at::randn({2, 2});
auto c = a + b.to(at::kInt);
```

上面 graph 中 `aten::add` 实现了如下的计算 `{aten::add, "${0} + ${2}*${1}"}`，也就是输入 0 加上输入 2 乘以输入 1。其中输入 1 为常数 1，所以结果为两个输入相加。

```
graph(%self.1 : __torch__.MyCell,
      %x : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu),
      %h : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu)):
  # 定义的 Linear 层，从 %self.1 获取属性 linear
  %linear : __torch__.torch.nn.modules.linear.Linear = prim::GetAttr[name="linear"](%self.1)
  # 调用 %linear 的前向函数，保存到 %20 中
  %20 : Tensor = prim::CallMethod[name="forward"](%linear, %x)
  # 常数 Tensor 1，保存到 %11 中
  %11 : int = prim::Constant[value=1]() # /path/to/test.py:9:0
  # 计算 %20 + %h * %11 保存到 %12 中
  %12 : Float(3, 4, strides=[4, 1], requires_grad=1, device=cpu) = aten::add(%20, %h, %11) # /path/to/test.py:9:0
  # 计算 tanh(%12) 保存到 %13 中
  %13 : Float(3, 4, strides=[4, 1], requires_grad=1, device=cpu) = aten::tanh(%12) # /path/to/test.py:9:0
  # 构建 tuple(%13, %13) 作为返回，保存到 %14 中
  %14 : (Float(3, 4, strides=[4, 1], requires_grad=1, device=cpu), Float(3, 4, strides=[4, 1], requires_grad=1, device=cpu)) = prim::TupleConstruct(%13, %13)
  return (%14)
```

## 控制流

增加一个控制流的子模块。

```
import torch

class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

class MyCell(torch.nn.Module):
    def __init__(self, dg):
        super(MyCell, self).__init__()
        self.dg = dg
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h

x, h = torch.rand(3, 4), torch.rand(3, 4)
my_cell = MyCell(MyDecisionGate())
traced_cell = torch.jit.trace(my_cell, (x, h))

print(traced_cell.dg.code)
print(traced_cell.code)

```

执行结果如下：

```
/path/to/control_flow.py:5: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if x.sum() > 0:
def forward(self,
    argument_1: Tensor) -> NoneType:
  return None

def forward(self,
    x: Tensor,
    h: Tensor) -> Tuple[Tensor, Tensor]:
  dg = self.dg
  linear = self.linear
  _0 = (linear).forward(x, )
  _1 = (dg).forward(_0, )
  _2 = torch.tanh(torch.add(_0, h))
  return (_2, _2)
```

从 `.code` 输出可以看出，`if-else` 分支不在其中，为什么呢？跟踪(tracing)确切地做了我们说的事情：运行代码，记录发生的操作并构造一个完全执行这些操作的 `ScriptModule`。不幸的是，诸如控制流之类的内容都被删除了。

### torch.jit.script

如何在 `TorchScript` 中表示这个模块呢？pytorch 提供了一个脚本编译器，它直接分析 Python 源代码，将其转换为 TorchScript。使用脚本编译器将 MyDecisionGate 转换为 TorchScript：

```
scripted_gate = torch.jit.script(MyDecisionGate())

my_cell = MyCell(scripted_gate)
scripted_cell = torch.jit.script(my_cell)

print(scripted_gate.code)
print(scripted_cell.code)
```
执行结果如下：
```
def forward(self,
    x: Tensor) -> Tensor:
  if bool(torch.gt(torch.sum(x), 0)):
    _0 = x
  else:
    _0 = torch.neg(x)
  return _0

def forward(self,
    x: Tensor,
    h: Tensor) -> Tuple[Tensor, Tensor]:
  dg = self.dg
  linear = self.linear
  _0 = torch.add((dg).forward((linear).forward(x, ), ), h)
  new_h = torch.tanh(_0)
  return (new_h, new_h)

```

## Scripting and Tracing 同时使用

有些情况需要使用 Tracing 而不是 Scripting（例如，一个模块有许多基于常量值做出的架构决策，我们希望这些值不会出现在 TorchScript 中）。在这种情况下，Scripting 可以与 Tracing 组合使用：torch.jit.script 会将 Tracing 模块的代码内联，而 Tracing 将编写的模块的代码内联。

```
import torch

class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

class MyCell(torch.nn.Module):
    def __init__(self, dg):
        super(MyCell, self).__init__()
        self.dg = dg
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h

scripted_gate = torch.jit.script(MyDecisionGate())

class MyRNNLoop(torch.nn.Module):
    def __init__(self):
        super(MyRNNLoop, self).__init__()
        x, h = torch.rand(3, 4), torch.rand(3, 4)
        self.cell = torch.jit.trace(MyCell(scripted_gate), (x, h))

    def forward(self, xs):
        h, y = torch.zeros(3, 4), torch.zeros(3, 4)
        for i in range(xs.size(0)):
            y, h = self.cell(xs[i], h)
        return y, h

rnn_loop = torch.jit.script(MyRNNLoop())
print("=== scripted_gate ===")
print(scripted_gate.code)
print("=== rnn_loop.cell ===")
print(rnn_loop.cell.code)
print("=== rnn_loop ===")
print(rnn_loop.code)
```
输出结果如下：

```
=== scripted_gate ===
def forward(self,
    x: Tensor) -> Tensor:
  if bool(torch.gt(torch.sum(x), 0)):
    _0 = x
  else:
    _0 = torch.neg(x)
  return _0

=== rnn_loop.cell ===
def forward(self,
    x: Tensor,
    h: Tensor) -> Tuple[Tensor, Tensor]:
  dg = self.dg
  linear = self.linear
  _0 = torch.add((dg).forward((linear).forward(x, ), ), h)
  _1 = torch.tanh(_0)
  return (_1, _1)

=== rnn_loop ===
def forward(self,
    xs: Tensor) -> Tuple[Tensor, Tensor]:
  h = torch.zeros([3, 4])
  y = torch.zeros([3, 4])
  y0 = y
  h0 = h
  for i in range(torch.size(xs, 0)):
    cell = self.cell
    _0 = (cell).forward(torch.select(xs, 0, i), h0, )
    y1, h1, = _0
    y0, h0 = y1, h1
  return (y0, h0)

```

## 参考文献
- https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html 
- https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html 
- https://pytorch.org/docs/stable/jit.html 
- https://pytorch.org/docs/master/jit_language_reference.html#language-reference
- https://pytorch.org/cppdocs/