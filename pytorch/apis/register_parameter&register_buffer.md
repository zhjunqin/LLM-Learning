# register_parameter 和 register_buffer 的区别

https://geek-docs.com/pytorch/pytorch-questions/50_pytorch_what_is_the_difference_between_register_parameter_and_register_buffer_in_pytorch.html
https://zhuanlan.zhihu.com/p/574259713

Pytorch 保存模型是保存状态字典（state_dict），也即 `torch.save(model.state_dict())`
但是有一个问题就是 model 的类成员变量不会被放入 model.state_dict() 被保存；
同样地，model 的类成员变量也不会随 model.cuda() 命令被复制到 GPU 中。
这就引出了两个问题，如何让模型需要的一些量被保存，以及如何将这些模型需要的量送入 GPU。

模型保存下来的参数有两种：一种是需要更新的 Parameter，另一种是不需要更新的 buffer。
在模型中，利用 backward 反向传播，可以通过 requires_grad 来得到 buffer 和 parameter 的梯度信息，但是利用 optimizer 进行更新的是 parameter，buffer 不会更新，这也是两者最重要的区别。
这两种参数都存在于 model.state_dict() 的 OrderedDict 中，也会随着模型“移动”（model.cuda()）。
对 Parameter 的访问 model.parameters() 或者 model.named_parameters()，对 buffer 的访问 model.buffers() 或者 model.named_buffers()。
可以看到，buffer 实际上是一种不被更新的 parameter，那么将 parameter 的 requires_grad 参数设置为 False 也可以达到和 buffer 类似的效果，但是对其他正常 parameter 的某些操作可能会影响到这些 "requires_grad==True" 的变量。

batch normalization 层中的 running_mean, running_var, num_batches_tracked 这三个参数是 buffer 类型的，这样既可以用 state_dict() 保存，也不会随着 optimizer 更新。

```
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(3, 3))
        self.register_parameter('bias', nn.Parameter(torch.Tensor(3)))
        self.register_buffer('running_mean', torch.zeros(3))
        self.register_buffer('running_var', torch.ones(3))

model = MyModel()
```

```
>>> model = MyModel()
>>> model.state_dict()
OrderedDict([
    ('weight', tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00],
        [1.3593e-43, 0.0000e+00, 4.0968e-35]])), 
    ('bias', tensor([7.6149e-20, 4.5712e-41, 4.1019e-35])), 
    ('running_mean', tensor([0., 0., 0.])), 
    ('running_var', tensor([1., 1., 1.]))])
>>> [i for i in model.parameters()]
[Parameter containing:
tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00],
        [1.3593e-43, 0.0000e+00, 4.0968e-35]], requires_grad=True), Parameter containing:
tensor([7.6149e-20, 4.5712e-41, 4.1019e-35], requires_grad=True)]
>>> [i for i in model.buffers()]
[tensor([0., 0., 0.]), tensor([1., 1., 1.])]
>>> [i for i in model.named_buffers()]
[('running_mean', tensor([0., 0., 0.])), ('running_var', tensor([1., 1., 1.]))]
```