# author: xiaodongguaAIGC
# KV-Cache + Generation + decoder

import torch
import torch.nn.functional as F
from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM
import math

D = 32  # single-head-dim
V = 64  # vocab_size


class Attention(torch.nn.Module):
    def __init__(self, D, V):
        super().__init__()
        self.D = D
        self.V = V
        self.Wq = torch.nn.Linear(D, D)
        self.Wk = torch.nn.Linear(D, D)
        self.Wv = torch.nn.Linear(D, D)
        self.cache_K = self.cache_V = None  # initial

    def forward(self, X, use_cache):
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)

        # Easy KV_Cache
        if use_cache:
            if self.cache_K == None:
                self.cache_K = K
                self.cache_V = V
            else:
                self.cache_K = torch.cat((self.cache_K, K), dim=1)
                self.cache_V = torch.cat((self.cache_V, V), dim=1)
                K = self.cache_K
                V = self.cache_V

            # print("cache_K:", self.cache_K.shape)
            # print("cache_V:", self.cache_K.shape)

        # ignore proj/MLP/scaled/mask/multi-head when calculate Attention
        # attn =Q@K.transpose(1,2)@V
        attn = (Q @ K.transpose(1, 2)) * (1.0 / math.sqrt(K.size(-1)))
        attn = F.softmax(attn, dim=-1)
        # print("After softmax", attn)
        attn = attn @ V
        # print("after @V", attn)
        return attn


class xiaodonggua_kv_cache(torch.nn.Module):
    def __init__(self, D, V):
        super().__init__()
        self.D = D
        self.V = V
        self.Embedding = torch.nn.Embedding(V, D)
        self.attention1 = Attention(D, V)
        self.attention2 = Attention(D, V)
        self.lm_head = torch.nn.Linear(D, V)  # LM_head

    def forward(self, X, use_cache):
        X = self.Embedding(X)
        X = self.attention1(X, use_cache)
        X = self.attention2(X, use_cache)

        output = self.lm_head(X)

        return output


model = xiaodonggua_kv_cache(D, V)

# 创建数据、不使用tokenizer
Z = torch.randint(0, 64, (1, 5))


print("=== no kv cache ===")
X = Z
print("input X", X.shape, X)
for i in range(3):
    # print(f"\nGeneration {i} step input_shape: {X.shape}：")
    print("==== Step {} ===".format(i))
    output = model.forward(X, False)
    # print(output.shape)
    next_token = torch.argmax(F.softmax(output, dim=-1), -1)[:, -1]
    # print(next_token.shape)
    x1 = next_token.unsqueeze(0)
    X = torch.cat((X, x1), dim=1)
    print("X.shape", X.shape, X)


print("=== no kv cache ===")
X = Z
print("input X", X.shape, X)
for i in range(3):
    # print(f"\nGeneration {i} step input_shape: {X.shape}：")
    print("==== Step {} ===".format(i))
    output = model.forward(X, True)
    # print(output.shape)
    next_token = torch.argmax(F.softmax(output, dim=-1), -1)[:, -1]
    # print(next_token.shape)
    X = next_token.unsqueeze(0)
    print("next X", X)
