# Challenge C: torch.compile for QLoRA - Research Summary

## Purpose
Research findings for implementing torch.compile without graph breaks for QLoRA fine-tuning.

---

## Scoring Criteria Analysis (Max 9 Points)

```python
if uses_flex_attention:
    if dynamic_sequence_length_works: C_score += 3
    else: C_score += 1
if no_torch_compile_BnB: C_score -= 2
elif use_part_A: C_score += 1
elif torch_compile_BnB: C_score += 1

if attention_compiled:
    if excessive_recompilation: C_score -= 3
    else: C_score += 2
if mlp_compiled:
    if excessive_recompilation: C_score -= 3
    C_score += 1

if not loss_compiled: C_score -= 1
if not layernorms_compiled: C_score -= 3

if max_autotune_triton_matmul:
    if excessive_recompilation: C_score -= 2
    else: C_score += 2
```

**Target Score Breakdown:**
- flex_attention with dynamic shapes: +3
- Use Part A kernel for BnB: +1
- Attention compiled (no excessive recompile): +2
- MLP compiled: +1
- Loss compiled: (avoid -1)
- LayerNorms compiled: (avoid -3)
- max_autotune matmul (no excessive recompile): +2
- **Potential max: 9 points**

---

## Key Research Findings

### 1. BitsAndBytes + torch.compile Incompatibility

**Problem:** BitsAndBytes doesn't support torch.compile natively. Graph breaks occur in `Linear4bit` layers.

**Source:** [PEFT Issue #1886](https://github.com/huggingface/peft/issues/1886) - "quantization does not work with torch.compile... bitsandbytes doesn't support torch.compile"

**Solutions:**
1. **Use Part A Triton kernel** - Replace bnb dequantization with our custom `your_dequantize_nf4` kernel (+1 point)
2. **Workaround** - Set `layer.compute_type_is_set = True` to skip compute type checking
3. **Wrap with torch._dynamo.disable** - Mark BnB operations as non-compilable

**Recommendation:** Use Part A kernel with proper torch.compile wrapping.

### 2. flex_attention for Dynamic Sequence Lengths

**Source:** [PyTorch Flex Attention Docs](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html)

**Key Pattern:**
```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

# Cache the block_mask - expensive to create
block_mask = create_block_mask(causal_mask, B, H, Q_LEN, KV_LEN, device="cuda")
output = flex_attention(query, key, value, block_mask=block_mask)
```

**Important Notes:**
- `create_block_mask` is expensive - must cache it
- Requires `torch.compile` for good performance
- Handles dynamic shapes better than SDPA
- For varying sequence lengths, use `causal_lower_right` pattern

**Dynamic Shape Handling:**
```python
# For variable sequence lengths in batches
def causal_lower_right(b, h, q_idx, kv_idx, q_len, kv_len):
    # Shift indices for right-aligned sequences
    return (q_len - q_idx) >= (kv_len - kv_idx)
```

### 3. Regional Compilation Strategy

**Source:** [PyTorch Regional Compilation Tutorial](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)

**Key Pattern:**
```python
class Model(nn.Module):
    def __init__(self):
        # Compile each repeated layer individually
        self.layers = nn.ModuleList([
            torch.compile(TransformerBlock()) for _ in range(num_layers)
        ])
```

**Benefits:**
- ~13x faster compilation vs full model
- Single graph reused for all layers
- For PyTorch < 2.5: `torch._dynamo.config.inline_inbuilt_nn_modules = True`

### 4. Compiled Autograd

**Source:** [PyTorch Compiled Autograd Tutorial](https://docs.pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html)

**Enable via:**
```python
# Method 1: Config flag
torch._dynamo.config.compiled_autograd = True

# Method 2: Context manager
with torch._dynamo.compiled_autograd.enable(torch.compile(fullgraph=True)):
    loss.backward()
```

**Benefits:**
- Captures full backward graph
- Graph breaks in forward don't break backward

### 5. torch.compile Options for QLoRA

**Recommended Settings:**
```python
torch_compile_options = {
    "epilogue_fusion": True,      # Fuse post-ops
    "max_autotune": True,         # Triton matmul autotuning (+2 points)
    "shape_padding": True,        # Pad shapes for better kernel selection
    "trace.enabled": True,        # Enable tracing
    "triton.cudagraphs": False,   # Disable for dynamic shapes
}

@torch.compile(fullgraph=False, dynamic=True, options=torch_compile_options)
def compiled_forward(...):
    ...
```

**Key Flags:**
- `fullgraph=False` - Allow graph breaks (for BnB operations)
- `dynamic=True` - Handle varying tensor shapes
- `max_autotune=True` - Required for +2 points

---

## Implementation Plan

### Phase 1: Patch BnB Linear4bit
1. Wrap `your_dequantize_nf4` from Part A as torch.compile-compatible
2. Replace `Linear4bit.forward` with compiled version using our kernel

### Phase 2: Patch Llama Components

**MLP (already in example):**
```python
@torch.compile(fullgraph=False, dynamic=True, options=torch_compile_options)
def compiled_llama_mlp(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj
```

**Attention with flex_attention:**
```python
@torch.compile(fullgraph=False, dynamic=True, options=torch_compile_options)
def compiled_llama_attention(self, hidden_states, ...):
    # Use flex_attention instead of SDPA
    Q = self.q_proj(hidden_states)
    K = self.k_proj(hidden_states)
    V = self.v_proj(hidden_states)

    # Apply rotary embeddings
    Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

    # Use flex_attention with cached block_mask
    attn_output = flex_attention(Q, K, V, block_mask=self._block_mask)
    return self.o_proj(attn_output)
```

**LayerNorms:**
```python
@torch.compile(fullgraph=True, dynamic=True)
def compiled_rms_norm(self, x):
    return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

**Loss:**
```python
@torch.compile(fullgraph=True, dynamic=True)
def compiled_cross_entropy(logits, labels):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
```

### Phase 3: Integration
1. Create patching functions for all components
2. Apply patches before model loading
3. Set up flex_attention block_mask caching
4. Enable compiled_autograd

---

## Expected Graph Breaks

With proper patching, expected compilations:
- 1 per unique layer type (MLP, Attention, LayerNorm)
- Reused across all layers via regional compilation
- Target: < 30 total compilations

**Avoid:**
- Recompiling for each sequence length change
- Recompiling for each layer
- Graph breaks from BnB (use Part A kernel)

---

## References

- [PEFT Issue #1886 - torch.compile + quantization](https://github.com/huggingface/peft/issues/1886)
- [bitsandbytes Issue #1184 - torch.compile support](https://github.com/TimDettmers/bitsandbytes/issues/1184)
- [PyTorch Flex Attention](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html)
- [PyTorch Regional Compilation](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)
- [PyTorch Compiled Autograd](https://docs.pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html)
- [PyTorch Forum - flex_attention slow training](https://discuss.pytorch.org/t/training-with-flex-attention-is-extremely-slow-due-to-torch-compile-settings/222581)
