---
layout: post
title:  "LLM Model Sharding with LLAMA3"
date:   2025-11-29
categories: [AI]
author: Alex Chiang
---

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-11-29-LLM-Model-Sharding-with-LLAMA3/llm_gpu_meme.png?raw=true" width="600" />
  </figure>
</div>

The biggest pain point of LLMs is not compute — it’s **MEMORY**.
The models are too large to run locally on a single consumer GPU.


## Why Model Sharding is Essential

LLMs often come with massive parameter sizes, for example, `LLAMA3-70B` is above 100GB in f16, that far exceed the memory capacity of a single consumer GPU.

Even with quantization techniques such as 8-bit or 4-bit, these models easily require ~40GB or more, which is still beyond what typical GPUs like the NVIDIA RTX 4080 (16GB), AMD RX 9070 (16GB) can hold.

> **NOTE** Quantized models often require hardware support for INT4/INT8/FP8 compute, and consumer GPUs do not always provide these capabilities.

This is where model sharding becomes essential. After sharding, the model weights are roughly reduced to `weights / device_count` per GPU.
Instead of requiring a single expensive datacenter GPU such as NVIDIA A100 or AMD SMI300, which are significantly more expensive than consumer GPUs, model sharding splits the large weights into multiple chunks and distributes them across several more affordable consumer GPUs.


## Matrix Multiplication with Tensor Parallelism (TP)

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-11-29-LLM-Model-Sharding-with-LLAMA3/a_b_c_0.png?raw=true" width="450" />
    <figcaption style="font-size: 14px; color: #555;">(image by author)</figcaption>
  </figure>
</div>

Before diving into model sharding, we need to know how could we compute matrix multiplication with tensor parallelism.
This section will explain how could we split 

$$
A * B = C
$$

### Pattern 1

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-11-29-LLM-Model-Sharding-with-LLAMA3/pattern_1.png?raw=true" width="450" />
    <figcaption style="font-size: 14px; color: #555;">(image by author)</figcaption>
  </figure>
</div>

$$
A * B = \text{Concat}(A * B_{0}, A * B_{1}) = \text{Concat}(C_{0}, C_{1}) = C
$$

We split B column-wise, compute each partial output independently, and then concatenate the results.

### Pattern 2

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-11-29-LLM-Model-Sharding-with-LLAMA3/pattern_2.png?raw=true" width="450" />
    <figcaption style="font-size: 14px; color: #555;">(image by author)</figcaption>
  </figure>
</div>

$$
A * B = (A_{0} * B_{0}) + (A_{1} * B_{1}) = C_{0} + C_{1} = C
$$

We split A row-wise and B column-wise, compute partial results on each shard, and then sum them (typically via all-reduce operation) to obtain the final output.

### Combining Both Patterns

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-11-29-LLM-Model-Sharding-with-LLAMA3/combined_pattern.png?raw=true" width="1200" />
    <figcaption style="font-size: 14px; color: #555;">(image by author)</figcaption>
  </figure>
</div>

By combining the two patterns above, we can perform one additional matrix multiplication that does not require synchronization across devices.


## Study Material - LLAMA3-8B

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-11-29-LLM-Model-Sharding-with-LLAMA3/llama3_arch.png?raw=true" width="400" />
    <figcaption style="font-size: 14px; color: #555;">(image by author)</figcaption>
  </figure>
</div>

In this section, we use `LLAMA3-8B` as an example to demonstrate how tensor parallelism (TP) can be applied across two devices. For simplicity, we assume a batch size of 1 and the input sequence length is `?`. As shown in the figure above, TP will be applied to `self-attention` and `FFN`.

The model parameters for `LLAMA3-8B` is as follows:
```txt
vocab_size      = 128256
hidden_dim      = 4096      (embedding_length)
ffn_dim         = 14336     (feed_forward_length)
layers          = 32        (block_count)
heads           = 32        (attention_head_count)
kv_heads        = 8         (attention_head_count_kv)
head_dim        = 128       (hidden_dim / num_heads = 4096 / 32)
kv_dim          = 1024      (kv_heads * head_dim = 8 * 128)
context_length  = 131072
// our assuming
batch_size      = 1
seq_len         = ?
```

After model sharding, the parameters will be:

|            | LLAMA3 (whole model) | Device:0 | Device:1 |
|------------|----------------------|----------|----------|
| batch_size | 1                    | 1        | 1        |
| hidden_dim | 4096                 | 4096     | 4096     |
| seq_len    | ?                    | ?        | ?        |
| heads      | 32                   | 16       | 16       |
| kv_heads   | 8                    | 4        | 4        |
| head_dim   | 128                  | 128      | 128      |
| ffn_dim    | 14336                | 7168     | 7168     |

The sharded weight shapes stored in GGUF:
```txt
// LLAMA3 (whole model)
// self-attention
// attn_q.weight: [heads * head_dim, hidden_dim]         = [4096, 4096]
// attn_k.weight: [kv_heads * head_dim, hidden_dim]      = [1024, 4096]
// attn_v.weight: [kv_heads * head_dim, hidden_dim]      = [1024, 4096]
// attn_output.weight: [hidden_dim, heads * head_dim]    = [4096, 4096]
// ffn
// ffn_up.weight: [ffn_dim, hidden_dim]                  = [14336, 4096]
// ffn_gate.weight: [ffn_dim, hidden_dim]                = [14336, 4096]
// ffn_down.weight: [hidden_dim, ffn_dim]                = [4096, 14336]

// Sharded: Device:0 / Device:1

attn_q.weight.sharded   = [2048, 4096] // W_q
attn_k.weight.sharded   = [512, 4096]  // W_k
attn_v.weight.sharded   = [512, 4096]  // W_v
attn_output.weight      = [4096, 2048] // W_o

ffn_up.weight.sharded   = [7168, 4096] // W_up
ffn_gate.weight.sharded = [7168, 4096] // W_gate
ffn_down.weight.sharded = [4096, 7168] // W_down
```

We will try to use the combined patterns to shard `self-attetion` and `FFN`.

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-11-29-LLM-Model-Sharding-with-LLAMA3/combined_patterns_sharded_model.png?raw=true" width="1200" />
    <figcaption style="font-size: 14px; color: #555;">(image by author)</figcaption>
  </figure>
</div>

In tensor parallelism, the `self-attention` and `FFN` layers are sharded across GPUs and require synchronization after computation. Lighter layers such as embeddings and normalization layers are fully replicated on each device.

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-11-29-LLM-Model-Sharding-with-LLAMA3/runtime.png?raw=true" width="400" />
    <figcaption style="font-size: 14px; color: #555;">(image by author)</figcaption>
  </figure>
</div>

Next, we will go through sharded `self-attention` and `FFN` layer to trace how the sharded weights are computed.

### self-attention

In this section, we walk through the `self-attention` layer and trace how the sharded weights are computed.

#### 1. QKV Projection

```txt
// Input tokens after embeding
x: [batch_size, seq_len, hidden_dim] = [1, ?, 4096]

// Linear: output = input @ weight^T
Q = x @ W_q ^ T = [1, ?, 4096] @ [4096, 4096] = [1, ?, 4096]
K = x @ W_k ^ T = [1, ?, 4096] @ [4096, 1024] = [1, ?, 1024]
V = x @ W_v ^ T = [1, ?, 4096] @ [4096, 1024] = [1, ?, 1024]
```

Model sharding in device:0: (device:1 is the same)
```txt
// Linear: output = input @ weight^T
Q = x @ W_q ^ T = [1, ?, 4096] @ [4096, 2048] = [1, ?, 2048]    /// pattern 1
K = x @ W_k ^ T = [1, ?, 4096] @ [4096, 512]  = [1, ?, 512]     /// pattern 1
V = x @ W_v ^ T = [1, ?, 4096] @ [4096, 512]  = [1, ?, 512]     /// pattern 1
```

#### 2. Position Encoding - RoPE

Skip the details because it won't affect tensor shapes.

#### 3. Reshape to Multi-Hea Format

```txt
// Transpose to [batch_size, heads, seq_len, head_dim]
Q = [1, ?, 4096] -> [1, 32, ?, 128]
K = [1, ?, 1024] -> [1, 8, ?, 128]
V = [1, ?, 1024] -> [1, 8, ?, 128]
```

Model sharding in device:0: (device:1 is the same)
```txt
Q = [1, ?, 2048] -> [1, 16, ?, 128]
K = [1, ?, 512]  -> [1, 4, ?, 128]
V = [1, ?, 512]  -> [1, 4, ?, 128]
```

#### 4. Group Query Attention (GQA) - Key/Value Expansion

```txt
// GQA: num_heads (32) > num_kv_heads (8)
// Each KV head is shared by (num_heads / num_kv_heads) = 4 query heads
// Expand K and V by repeating each head 4 times

groups = num_heads / num_kv_heads = 32 / 8 = 4

K = [1, 8, ?, 128] -> [1, 32, ?, 128]
V = [1, 8, ?, 128] -> [1, 32, ?, 128]
```

Model sharding in device:0: (device:1 is the same)
```txt
K = [1, 4, ?, 128] -> [1, 16, ?, 128]
V = [1, 4, ?, 128] -> [1, 16, ?, 128]
```

#### 5. Attention Computation

$$
\text{Score} = \text{Softmax}((Q * K^T)) / \sqrt(d) * V
$$

```txt
// Scaled dot-product attention
// scores = (Q @ K^T) / sqrt(head_dim)

scores = Q @ K^T
       = [1, 32, ?, 128] @ [1, 32, 128, ?]
       = [1, 32, ?, ?]

scores = scores / sqrt(head_dim)
       = scores / sqrt(128)
       = scores / 11.314
       = [1, 32, ?, ?]

// Causal masking
// Skip the details becasue it won't affect the tensor shapes
scores = [1, 32, ?, ?]

// Softmax
attn_weights = softmax(scores, dim=-1) = [1, 32, ?, ?]

// Apply attention to values
attn_output = attn_weights @ V 
            = [1, 32, ?, ?] @ [1, 32, ?, 128]
            = [1, 32, ?, 128]

// Reshape - concatenate all heads
attn_output = [1, 32, ?, 128] -> [1, ?, 4096]

// Output projection
attn_output = attn_output @ W_o^T
            = [1, ?, 4096] @ [4096, 4096]^T
            = [1, ?, 4096]
```

Model sharding in device:0: (device:1 is the same)
```txt
scores = Q @ K^T
       = [1, 16, ?, 128] @ [1, 16, 128, ?]
       = [1, 16, ?, ?]
scores = scores / sqrt(head_dim)

// Softmax
attn_weights = softmax(scores, dim=-1) = [1, 16, ?, ?]

// Apply attention to values
attn_output = attn_weights @ V 
            = [1, 16, ?, ?] @ [1, 16, ?, 128]
            = [1, 16, ?, 128]

// Reshape - concatenate all heads
attn_output = [1, 16, ?, 128] -> [1, ?, 2048]

// Output projection
attn_output = attn_output @ W_o^T
            = [1, ?, 2048] @ [4096, 2048]^T    /// pattern 2
            = [1, ?, 4096]
```

You may notice that **tensor parallelism in LLMs typically involves distributing attention heads across devices**. The following is the visualization of the attention computation.

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-11-29-LLM-Model-Sharding-with-LLAMA3/vis_attn.png?raw=true" width="800" />
    <figcaption style="font-size: 14px; color: #555;">(image by author)</figcaption>
  </figure>
</div>

#### 6. Synchronization (all-reduce)

Finally, we need to synchronize the results from both devices, typically using an all-reduce operation.

### FFN

In this section, we walk through the `FFN` layer and trace how the sharded weights are computed.

```txt
// ffn_gate.weight: [ffn_dim, hidden_dim] = [14336, 4096]
// ffn_up.weight:   [ffn_dim, hidden_dim] = [14336, 4096]

gate = x @ W_gate^T = [1, ?, 4096] @ [4096, 14336].T = [1, ?, 14336]
up   = x @ W_up^T   = [1, ?, 4096] @ [4096, 14336].T = [1, ?, 14336]

// SwiGLU: silu(gate) * up
// silu(x) = x * sigmoid(x)

gate_activated = silu(gate) = gate * sigmoid(gate)
               = [1, ?, 14336]

ffn_hidden = gate_activated * up
           = [1, ?, 14336]

// ffn_down.weight: [hidden_dim, ffn_dim] = [4096, 14336]

ffn_output = ffn_hidden @ W_down^T
           = [1, ?, 14336] @ [14336, 4096]^T
           = [1, ?, 4096]
```

Model sharding in device:0: (device:1 is the same)
```txt
gate = x @ W_gate^T = [1, ?, 4096] @ [4096, 7168].T = [1, ?, 7168]    /// pattern 1
up   = x @ W_up^T   = [1, ?, 4096] @ [4096, 7168].T = [1, ?, 7168]    /// pattern 1

// SwiGLU: silu(gate) * up
// silu(x) = x * sigmoid(x)

gate_activated = silu(gate) = gate * sigmoid(gate)
               = [1, ?, 7168]

ffn_hidden = gate_activated * up
           = [1, ?, 7168]

// ffn_down.weight: [hidden_dim, ffn_dim] = [4096, 7168]

ffn_output = ffn_hidden @ W_down^T
           = [1, ?, 7168] @ [7168, 4096]^T    /// pattern 2
           = [1, ?, 4096]
```

Finally, we need to synchronize the results from both devices, typically using an all-reduce operation.


## The Other Sharding Types

While tensor parallelism (TP) focuses on sharding the large matrix multiplications within each layer, two additional strategies—Pipeline Parallelism (PP) and Data Parallelism (DP)—are commonly used alongside TP to scale large models across multiple GPUs or nodes.

- Pipeline Parallelism (PP): If each device can hold an entire layer, different layers of the model are assigned to different devices. For example, device 0 runs layers 0–3, while device 1 runs layers 4–7. Activations must be transferred between stages.
- Data Parallelism (DP): If each device can hold the entire model, the full model is replicated on every device, and each device processes a different batch of data. Afterward, the batch results need to be combined (e.g., concatenated or reduced).

| Strategy | What's Split              | Communication                                                |
|----------|---------------------------|--------------------------------------------------------------|
| TP       | Weights                   | All-reduce activations                                       |
| DP       | Batch                     | Gather (inference) / All-reduce gradients (training)         |
| PP       | Layers                    | Transfer activations between stages                          |

In practice, all of these sharding strategies can be combined into what is known as 3D parallelism.
The following figure illustrates the attention computation under 3D parallelism.

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-11-29-LLM-Model-Sharding-with-LLAMA3/3d_parallelism.png?raw=true" width="800" />
    <figcaption style="font-size: 14px; color: #555;">(image by author)</figcaption>
  </figure>
</div>

We can assign each block to a different device, as shown in the following diagram.

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-11-29-LLM-Model-Sharding-with-LLAMA3/3d_parallelism_devices.png?raw=true" width="400" height="400"/>
    <figcaption style="font-size: 14px; color: #555;">(image by author)</figcaption>
  </figure>
</div>

## References

- [LLAMA3](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [Model Parallelism](https://huggingface.co/transformers/v4.10.1/parallelism.html)
- [shark ai](https://github.com/nod-ai/amd-shark-ai)
