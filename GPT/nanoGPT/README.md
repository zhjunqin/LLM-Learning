

# nano GPT

https://github.com/karpathy/nanoGPT

## 主要模型结构代码

```PYTHON
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # word token embedding (vocab_size, n_embd)
            wpe = nn.Embedding(config.block_size, config.n_embd), # word position embedding (block_size, n_embd)
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
```

## weight-tying

[Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859v3)

也就是输出的 embedding 矩阵和输入的 word token embedding 矩阵共享同一个矩阵。

torch.nn.Linear 中的权重的形状为 `(output_dim, input_dim)`：

$$y=xA^T + b$$

```
>>> from torch import nn
>>> import torch
>>> m = nn.Linear(20, 30)
>>> input = torch.randn(128, 20)
>>> output = m(input)
>>> print(output.size())
torch.Size([128, 30])
>>> m.weight.shape
torch.Size([30, 20])
```

## scaled dot product attention

[SCALED_DOT_PRODUCT_ATTENTION](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
目前有三种支持的 scaled dot product attention 实现：

- FlashAttention：具有 IO 感知能力的快速和内存高效的精确注意力

- Memory-Efficient Attention

- 一个用 C++ 定义的 PyTorch 实现

当使用 CUDA 后端时，该函数可以调用优化的内核以提高性能。对于所有其他后端，将使用 PyTorch 实现。

所有实现默认启用，scaled dot product attention 会尝试根据输入自动选择最优实现。

## Causal SelfAttention

```
causal adj 造成…的因果关系
Is it possible to trace the causal factors of the current crisis?

不同于 casual 随便，漫不经心
```

定义一个下三角的矩阵，形状为 (1, 1, block_size, block_size)
```
# causal mask to ensure that attention is only applied to the left in the input sequence
self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))
```

示例：
```
>>> block_size = 6
>>> torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
tensor([[[[1., 0., 0., 0., 0., 0.],
          [1., 1., 0., 0., 0., 0.],
          [1., 1., 1., 0., 0., 0.],
          [1., 1., 1., 1., 0., 0.],
          [1., 1., 1., 1., 1., 0.],
          [1., 1., 1., 1., 1., 1.]]]])
```

将 Tensor 中 mask 为 True 的元素填充为 `float('-inf')`
其中 `bias[:,:,:T,:T]` 将其截断到输入的序列长度，然后判断 att 中对应 `self.bias[:,:,:T,:T] == 0` 中为零的部分，填充为 `float('-inf')`，
这样后面 softmax 中这部分的值就会 0

```
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
att = F.softmax(att, dim=-1)
```

示例：

```
>>> from torch import nn
>>> import torch
>>> from torch.nn import functional as F
>>> block_size = 6
>>> torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
tensor([[[[1., 0., 0., 0., 0., 0.],
          [1., 1., 0., 0., 0., 0.],
          [1., 1., 1., 0., 0., 0.],
          [1., 1., 1., 1., 0., 0.],
          [1., 1., 1., 1., 1., 0.],
          [1., 1., 1., 1., 1., 1.]]]])
>>> bias = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
>>> att = torch.randn(2, 4, block_size, block_size)
>>> att.masked_fill(bias[:,:,:block_size,:block_size] == 0, float('-inf'))
tensor([[[[-1.1114,    -inf,    -inf,    -inf,    -inf,    -inf],
          [ 0.7620, -0.0800,    -inf,    -inf,    -inf,    -inf],
          [-0.0578, -1.2562, -0.9079,    -inf,    -inf,    -inf],
          [-1.0784, -1.1073, -0.1795, -0.5894,    -inf,    -inf],
          [ 1.0281, -0.1624, -0.9736, -0.7253, -0.8998,    -inf],
          [-1.4249,  0.6308, -1.9127,  0.2854, -0.4818, -0.1031]],

         [[-0.9006,    -inf,    -inf,    -inf,    -inf,    -inf],
          [ 0.1169, -0.3118,    -inf,    -inf,    -inf,    -inf],
          [-0.8000, -0.8861, -0.5884,    -inf,    -inf,    -inf],
          [-0.4021, -2.8034, -0.9937, -0.6041,    -inf,    -inf],
          [-1.2087, -0.6230,  0.2791, -1.2123, -1.9938,    -inf],
          [ 0.9249, -0.5817, -1.4905,  0.3995, -0.2248, -1.0676]],

         [[-0.0934,    -inf,    -inf,    -inf,    -inf,    -inf],
          [-0.8740,  1.8302,    -inf,    -inf,    -inf,    -inf],
          [-1.5717, -0.0799,  1.2241,    -inf,    -inf,    -inf],
          [-0.7450,  0.2566, -0.2840,  1.7136,    -inf,    -inf],
          [-1.8018,  0.8039,  0.8916, -0.2933,  1.0090,    -inf],
          [-0.7378, -1.2816, -0.4052, -1.1524, -1.4675,  1.5224]],

         [[-0.5070,    -inf,    -inf,    -inf,    -inf,    -inf],
          [ 0.7727,  0.6143,    -inf,    -inf,    -inf,    -inf],
          [ 0.3514,  1.6755,  0.2576,    -inf,    -inf,    -inf],
          [ 0.3869, -0.4979,  0.0099,  0.1388,    -inf,    -inf],
          [-0.3926, -0.0274, -0.7083, -1.1898,  0.2952,    -inf],
          [-0.0722,  1.4160, -0.7086, -0.5166,  1.3232,  0.4450]]],


        [[[ 0.7661,    -inf,    -inf,    -inf,    -inf,    -inf],
          [ 0.4473,  1.3872,    -inf,    -inf,    -inf,    -inf],
          [ 0.1208, -1.0697,  0.8647,    -inf,    -inf,    -inf],
          [-1.7977,  0.9678,  1.3069,  1.3894,    -inf,    -inf],
          [ 2.3455, -1.3579,  0.6791, -0.2250,  0.0589,    -inf],
          [-1.0527,  0.9221, -1.2351, -0.4458, -1.5264,  1.5150]],

         [[ 0.3839,    -inf,    -inf,    -inf,    -inf,    -inf],
          [ 1.6320,  2.2736,    -inf,    -inf,    -inf,    -inf],
          [ 1.1779, -0.5750, -0.5107,    -inf,    -inf,    -inf],
          [-0.5612,  0.5175,  1.9069, -0.5858,    -inf,    -inf],
          [ 0.4242,  0.1805, -2.0838,  1.9375,  2.0375,    -inf],
          [ 1.1747, -2.7261, -0.9377, -2.0122, -0.2254,  0.9096]],

         [[-0.5892,    -inf,    -inf,    -inf,    -inf,    -inf],
          [ 0.7585,  1.0200,    -inf,    -inf,    -inf,    -inf],
          [ 0.7357,  2.4511,  0.5434,    -inf,    -inf,    -inf],
          [ 0.1705,  0.9652, -2.0732,  2.1393,    -inf,    -inf],
          [-0.6041, -1.0505, -0.3397, -0.7604,  0.6324,    -inf],
          [ 2.2278, -0.1635,  2.1337, -1.8201, -1.6238,  1.1392]],

         [[ 1.2250,    -inf,    -inf,    -inf,    -inf,    -inf],
          [-0.8184,  0.7195,    -inf,    -inf,    -inf,    -inf],
          [ 0.4113, -0.9320, -0.5485,    -inf,    -inf,    -inf],
          [-0.8884, -0.9454,  1.4445, -0.8613,    -inf,    -inf],
          [ 1.1017,  1.5980, -0.2744, -0.3679, -0.2068,    -inf],
          [ 0.6375,  0.0060, -1.0044,  0.2665, -0.3344, -0.2008]]]])
>>> att = att.masked_fill(bias[:,:,:block_size,:block_size] == 0, float('-inf'))
>>> F.softmax(att, dim=-1)
tensor([[[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.6989, 0.3011, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.5784, 0.1745, 0.2472, 0.0000, 0.0000, 0.0000],
          [0.1650, 0.1603, 0.4055, 0.2691, 0.0000, 0.0000],
          [0.5689, 0.1730, 0.0769, 0.0985, 0.0827, 0.0000],
          [0.0470, 0.3672, 0.0289, 0.2600, 0.1207, 0.1763]],

         [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.6056, 0.3944, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.3171, 0.2910, 0.3919, 0.0000, 0.0000, 0.0000],
          [0.4063, 0.0368, 0.2249, 0.3320, 0.0000, 0.0000],
          [0.1153, 0.2070, 0.5103, 0.1148, 0.0526, 0.0000],
          [0.4245, 0.0941, 0.0379, 0.2510, 0.1345, 0.0579]],

         [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.0627, 0.9373, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.0458, 0.2037, 0.7505, 0.0000, 0.0000, 0.0000],
          [0.0588, 0.1602, 0.0933, 0.6877, 0.0000, 0.0000],
          [0.0198, 0.2683, 0.2929, 0.0896, 0.3294, 0.0000],
          [0.0730, 0.0424, 0.1018, 0.0482, 0.0352, 0.6995]],

         [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.5395, 0.4605, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.1764, 0.6630, 0.1606, 0.0000, 0.0000, 0.0000],
          [0.3473, 0.1434, 0.2383, 0.2710, 0.0000, 0.0000],
          [0.1783, 0.2568, 0.1300, 0.0803, 0.3546, 0.0000],
          [0.0812, 0.3597, 0.0430, 0.0521, 0.3278, 0.1362]]],


        [[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.2809, 0.7191, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.2934, 0.0892, 0.6174, 0.0000, 0.0000, 0.0000],
          [0.0158, 0.2506, 0.3517, 0.3820, 0.0000, 0.0000],
          [0.7186, 0.0177, 0.1358, 0.0550, 0.0730, 0.0000],
          [0.0408, 0.2937, 0.0340, 0.0748, 0.0254, 0.5314]],

         [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.3449, 0.6551, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.7363, 0.1276, 0.1361, 0.0000, 0.0000, 0.0000],
          [0.0598, 0.1759, 0.7059, 0.0584, 0.0000, 0.0000],
          [0.0875, 0.0686, 0.0071, 0.3975, 0.4393, 0.0000],
          [0.4553, 0.0092, 0.0551, 0.0188, 0.1123, 0.3493]],

         [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.4350, 0.5650, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.1354, 0.7528, 0.1117, 0.0000, 0.0000, 0.0000],
          [0.0954, 0.2112, 0.0101, 0.6833, 0.0000, 0.0000],
          [0.1381, 0.0884, 0.1799, 0.1181, 0.4755, 0.0000],
          [0.4207, 0.0385, 0.3829, 0.0073, 0.0089, 0.1416]],

         [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.1768, 0.8232, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.6083, 0.1587, 0.2330, 0.0000, 0.0000, 0.0000],
          [0.0753, 0.0711, 0.7762, 0.0774, 0.0000, 0.0000],
          [0.2945, 0.4838, 0.0744, 0.0677, 0.0796, 0.0000],
          [0.3100, 0.1648, 0.0600, 0.2139, 0.1173, 0.1340]]]])
```

## GELU

[GELU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html)

应用高斯误差线性单元函数（Gaussian Error Linear Units）：

$$GELU(x)=x∗Φ(x)$$

其中，$Φ(x)$ 是高斯分布的累积分布函数（Cumulative Distribution Function）。

当近似参数为 'tanh' 时，Gelu 使用以下估算式：

$$GELU(x)=0.5∗x∗(1+Tanh(\sqrt{2/π}∗(x+0.044715∗x^3)))$$

![](./GELU.png)

## 推理最后的输出

将 x 的第二维只取预测序列的最后一个

```
# x.shape torch.Size([1, 6, 384])
x = self.transformer.ln_f(x) 
# inference-time mini-optimization: only forward the lm_head on the very last position
# logits.shape torch.Size([1, 1, 65])
logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
```

## Temperature

[Temperature 在模型中的作用](https://avoid.overfit.cn/post/04f2376489184f53a6ae9c5d4b43dc97)

Temperature 是一个超参数，可用于控制生成语言模型中生成文本的随机性和创造性。它用于调整模型的 softmax 输出层中预测词的概率。Temperature 参数定义为在应用 softmax 函数之前用于调整 logits 的比例因子的倒数。

当 Temperature 设置为较低的值时，预测词的概率会变尖锐，这意味着选择最有可能的词的概率更高。这会产生更保守和可预测的文本，因为模型不太可能生成意想不到或不寻常的词。
另一方面，当 Temperature 设置为较高值时，预测词的概率被拉平，这意味着所有词被选择的可能性更大。这会产生更有创意和多样化的文本，因为模型更有可能生成不寻常或意想不到的词。

Temperature 参数通常设置为 0.1 到 1.0 之间的值，具体取决于生成文本中所需的随机性和创造性水平。温度值为 1.0 对应于标准 softmax 函数，其中预测词的概率未按比例缩放。

示例：
```
Prompt: "The quick brown fox"

Temperature = 0.1:

"The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog."

Temperature = 0.5:

"The quick brown fox jumped over the lazy dog. The lazy cat was not impressed. The quick brown fox ran away."

Temperature = 1.0:

"The quick brown fox jumped over the lazy dog. Suddenly, a flock of birds flew overhead, causing the fox to stop in its tracks. It looked up at the sky, wondering where they were going."

```

可以看到，Temperature 对生成文本的质量和创造性有重大影响。低值生成更可预测和重复的文本，而高值生成更多样化和创造性的文本。

Temperature 的数学原理解释

$$p(x_i) = \frac{e^{x_i}}{\sum e^{x_i}}$$

$$p(x_i) = \frac{e^{\frac{x_i}{T}}}{\sum e^{\frac{x_j}{T}}}$$

更深入的解释 Temperature 参数：

如果当 T 趋于无穷时会发生什么。每个 $\frac{x_i}{T}$ 都会趋于 0，从而得到一个均匀分布。也就是说概率分布变得更 “平”， 这会导致结果更随机。

当 T 很小(比如 0.1)时会发生什么。每个 $\frac{x_i}{T}$ 之间的差异变得更加明显，这样概率分布变得“更尖”，也就是说结果会更确定。


## multinomial

[multinomial](https://pytorch.org/docs/stable/generated/torch.multinomial.html)

返回一个张量，其中每一行包含从张量输入对应行的多项式概率分布中抽样得到的 num_samples 个索引。
```
probs = F.softmax(logits, dim=-1)
# sample from the distribution
idx_next = torch.multinomial(probs, num_samples=1)
# append sampled index to the running sequence and continue
idx = torch.cat((idx, idx_next), dim=1)
```

```
(Pdb) c
-> probs = F.softmax(logits, dim=-1)
(Pdb) p logits
tensor([[19.5000,  5.4688, -1.0625, -5.5625, -5.2500,  5.3750, -0.4277,  1.8047,
         -0.2148, -0.4766,  2.5156, -0.7734, -0.9844,  0.0850,  0.0635,  1.0703,
         -2.0938, -0.6992, -2.0938,  0.7969,  0.5781,  0.7617, -3.0312,  0.6211,
         -0.4570,  1.9531,  1.5547, -0.0933,  1.1016, -3.9062, -1.3594,  1.7812,
         -0.8516, -0.5781,  1.7891,  3.3594, -4.5625,  1.4141,  1.4922, -1.8047,
         -3.0312, -2.0312, -0.5273, -0.6016, -2.9688, -2.6562, -2.9062, -1.0781,
         -6.8438, -0.2910, -1.7578, -0.8438, -1.8672, -3.3125, -0.6328, -3.9375,
         -2.9062, -1.1328, -3.6094, -0.0786, -2.1094, -0.7578, -2.9688,  0.7500,
         -6.4062]], device='cuda:0', dtype=torch.bfloat16)
(Pdb) p logits.shape
torch.Size([1, 65])
(Pdb) n
-> idx_next = torch.multinomial(probs, num_samples=1)
(Pdb) p probs
tensor([[1.0000e+00, 8.0594e-07, 1.1744e-09, 1.3046e-11, 1.7832e-11, 7.3382e-07,
         2.2156e-09, 2.0655e-08, 2.7413e-09, 2.1100e-09, 4.2051e-08, 1.5680e-09,
         1.2698e-09, 3.6996e-09, 3.6210e-09, 9.9103e-09, 4.1875e-10, 1.6888e-09,
         4.1875e-10, 7.5394e-09, 6.0581e-09, 7.2789e-09, 1.6398e-10, 6.3240e-09,
         2.1516e-09, 2.3960e-08, 1.6086e-08, 3.0957e-09, 1.0225e-08, 6.8359e-11,
         8.7275e-10, 2.0176e-08, 1.4502e-09, 1.9063e-09, 2.0335e-08, 9.7772e-08,
         3.5464e-11, 1.3976e-08, 1.5111e-08, 5.5910e-10, 1.6398e-10, 4.4575e-10,
         2.0056e-09, 1.8621e-09, 1.7456e-10, 2.3860e-10, 1.8582e-10, 1.1562e-09,
         3.6229e-12, 2.5402e-09, 5.8593e-10, 1.4616e-09, 5.2523e-10, 1.2378e-10,
         1.8048e-09, 6.6256e-11, 1.8582e-10, 1.0947e-09, 9.1987e-11, 3.1413e-09,
         4.1226e-10, 1.5927e-09, 1.7456e-10, 7.1941e-09, 5.6112e-12]],
       device='cuda:0')
(Pdb) n
-> idx = torch.cat((idx, idx_next), dim=1)
(Pdb) p idx_next
tensor([[0]], device='cuda:0')
(Pdb) p idx_next.shape
torch.Size([1, 1])
(Pdb) p idx
tensor([[ 0,  0, 13, 26, 19, 17, 24, 27, 10]], device='cuda:0')
(Pdb) p idx.shape
torch.Size([1, 9])
(Pdb) p idx
tensor([[ 0,  0, 13, 26, 19, 17, 24, 27, 10,  0]], device='cuda:0')
```

## 前向过程

```
GPT(
  (transformer): ModuleDict(
    (wte): Embedding(65, 384)
    (wpe): Embedding(256, 384)
    (drop): Dropout(p=0.2, inplace=False)
    (h): ModuleList(
      (0-5): 6 x Block(
        (ln_1): LayerNorm()
        (attn): CausalSelfAttention(
          (c_attn): Linear(in_features=384, out_features=1152, bias=False) # 1152 = 3 * 384
          (c_proj): Linear(in_features=384, out_features=384, bias=False)
          (attn_dropout): Dropout(p=0.2, inplace=False)
          (resid_dropout): Dropout(p=0.2, inplace=False)
        )
        (ln_2): LayerNorm()
        (mlp): MLP(
          (c_fc): Linear(in_features=384, out_features=1536, bias=False) # 1536 = 4 * 384
          (gelu): GELU(approximate='none')
          (c_proj): Linear(in_features=1536, out_features=384, bias=False)
          (dropout): Dropout(p=0.2, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm()
  )
  (lm_head): Linear(in_features=384, out_features=65, bias=False)
)

```


这里介绍一下，整个前向过程的 shape 

```
# forward the GPT model itself
tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
x = self.transformer.drop(tok_emb + pos_emb) # (b, t, n_embd)
for block in self.transformer.h:
    x = block(x) # (b, t, n_embd)
x = self.transformer.ln_f(x) # (b, t, n_embd)

if targets is not None:
    # if we are given some desired targets also calculate the loss
    logits = self.lm_head(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
else:
    # inference-time mini-optimization: only forward the lm_head on the very last position
    logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim (b, 1, vocab_size)
```

## 统计模型的参数


```
total = 0
for name,parameters in model.named_parameters():
    print(name, ':', parameters.size(), parameters.numel())
    total += parameters.numel()
```

```
number of parameters: 10.65M
transformer.wte.weight :  torch.Size([65, 384]) :  24960
transformer.wpe.weight :  torch.Size([256, 384]) :  98304
transformer.h.0.ln_1.weight :  torch.Size([384]) :  384
transformer.h.0.attn.c_attn.weight :  torch.Size([1152, 384]) :  442368
transformer.h.0.attn.c_proj.weight :  torch.Size([384, 384]) :  147456
transformer.h.0.ln_2.weight :  torch.Size([384]) :  384
transformer.h.0.mlp.c_fc.weight :  torch.Size([1536, 384]) :  589824
transformer.h.0.mlp.c_proj.weight :  torch.Size([384, 1536]) :  589824
transformer.h.1.ln_1.weight :  torch.Size([384]) :  384
transformer.h.1.attn.c_attn.weight :  torch.Size([1152, 384]) :  442368
transformer.h.1.attn.c_proj.weight :  torch.Size([384, 384]) :  147456
transformer.h.1.ln_2.weight :  torch.Size([384]) :  384
transformer.h.1.mlp.c_fc.weight :  torch.Size([1536, 384]) :  589824
transformer.h.1.mlp.c_proj.weight :  torch.Size([384, 1536]) :  589824
transformer.h.2.ln_1.weight :  torch.Size([384]) :  384
transformer.h.2.attn.c_attn.weight :  torch.Size([1152, 384]) :  442368
transformer.h.2.attn.c_proj.weight :  torch.Size([384, 384]) :  147456
transformer.h.2.ln_2.weight :  torch.Size([384]) :  384
transformer.h.2.mlp.c_fc.weight :  torch.Size([1536, 384]) :  589824
transformer.h.2.mlp.c_proj.weight :  torch.Size([384, 1536]) :  589824
transformer.h.3.ln_1.weight :  torch.Size([384]) :  384
transformer.h.3.attn.c_attn.weight :  torch.Size([1152, 384]) :  442368
transformer.h.3.attn.c_proj.weight :  torch.Size([384, 384]) :  147456
transformer.h.3.ln_2.weight :  torch.Size([384]) :  384
transformer.h.3.mlp.c_fc.weight :  torch.Size([1536, 384]) :  589824
transformer.h.3.mlp.c_proj.weight :  torch.Size([384, 1536]) :  589824
transformer.h.4.ln_1.weight :  torch.Size([384]) :  384
transformer.h.4.attn.c_attn.weight :  torch.Size([1152, 384]) :  442368
transformer.h.4.attn.c_proj.weight :  torch.Size([384, 384]) :  147456
transformer.h.4.ln_2.weight :  torch.Size([384]) :  384
transformer.h.4.mlp.c_fc.weight :  torch.Size([1536, 384]) :  589824
transformer.h.4.mlp.c_proj.weight :  torch.Size([384, 1536]) :  589824
transformer.h.5.ln_1.weight :  torch.Size([384]) :  384
transformer.h.5.attn.c_attn.weight :  torch.Size([1152, 384]) :  442368
transformer.h.5.attn.c_proj.weight :  torch.Size([384, 384]) :  147456
transformer.h.5.ln_2.weight :  torch.Size([384]) :  384
transformer.h.5.mlp.c_fc.weight :  torch.Size([1536, 384]) :  589824
transformer.h.5.mlp.c_proj.weight :  torch.Size([384, 1536]) :  589824
transformer.ln_f.weight :  torch.Size([384]) :  384
total:  10,745,088
```

每个 Attention

```
transformer.h.0.ln_1.weight :  torch.Size([384]) :  384
transformer.h.0.attn.c_attn.weight :  torch.Size([1152, 384]) :  442368
transformer.h.0.attn.c_proj.weight :  torch.Size([384, 384]) :  147456
transformer.h.0.ln_2.weight :  torch.Size([384]) :  384
transformer.h.0.mlp.c_fc.weight :  torch.Size([1536, 384]) :  589824
transformer.h.0.mlp.c_proj.weight :  torch.Size([384, 1536]) :  589824

total: 1,770,240
```