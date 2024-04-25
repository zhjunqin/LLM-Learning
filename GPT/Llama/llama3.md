# Llama3

https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3

```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)

```


https://github.com/meta-llama/llama-recipes

> Llama 3 has a new prompt template and special tokens (based on the tiktoken tokenizer).
> | Token | Description |
> |---|---|
> `<\|begin_of_text\|>` | This is equivalent to the BOS token. |
> `<\|end_of_text\|>` | This is equivalent to the EOS token. For multiturn-conversations it's usually unused. Instead, every message is terminated with `<\|eot_id\|>` instead.|
> `<\|eot_id\|>` | This token signifies the end of the message in a turn i.e. the end of a single message by a system, user or assistant role as shown below.|
> `<\|start_header_id\|>{role}<\|end_header_id\|>` | These tokens enclose the role for a particular message. The possible roles can be: system, user, assistant. |
>
> A multiturn-conversation with Llama 3 follows this prompt template:
> ```
> <|begin_of_text|><|start_header_id|>system<|end_header_id|>
>
> {{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>
>
> {{ user_message_1 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
>
> {{ model_answer_1 }}<|eot_id|><|start_header_id|>user<|end_header_id|>
>
> {{ user_message_2 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
> ```
> Each message gets trailed by an `<|eot_id|>` token before a new header is started, signaling a role change.
>
