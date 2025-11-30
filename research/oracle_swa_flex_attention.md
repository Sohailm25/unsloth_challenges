# Oracle Research: Sliding Window Attention with flex_attention

**Date**: 2025-11-29
**Model**: gpt-5-pro
**Cost**: $1.9751

## Key Findings

### 1. SWA mask_mod Function

**Bidirectional (encoder-style):**
```python
def sliding_window(window_size: int):
    W = int(window_size)
    def _mask(b, h, q_idx, kv_idx):
        return (q_idx - kv_idx).abs() <= W
    return _mask
```

**Decoder-style (causal + window):**
```python
def one_sided_window(window_size: int):
    W = int(window_size)
    def _mask(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= W)
    return _mask
```

### 2. Composing Masks with and_masks

```python
from torch.nn.attention.flex_attention import and_masks

swa_causal = and_masks(causal, sliding_window(window_size))
```

### 3. SWA + Document Boundary Masking

```python
def make_doc_mask(doc_id: torch.Tensor):
    def _doc(b, h, q_idx, kv_idx):
        return doc_id[b, q_idx] == doc_id[b, kv_idx]
    return _doc

doc_mask = make_doc_mask(doc_id)
swa_mask = sliding_window(window_size)
causal_mask = causal
combo_mask = and_masks(causal_mask, swa_mask, doc_mask)
```

### 4. Performance Considerations

- **Prefer mask_mod + create_block_mask** over score_mod for masking
- **Compose masks with and_masks/or_masks** rather than branching inside mask_mod
- **Cache BlockMask objects** per (S, window_size, pattern) - avoid recreating each step
- **Don't change mask_mod logic or window_size** between iterations - causes recompiles
- **kernel_options["ROWS_GUARANTEED_SAFE"]=True** valid for causal+window (guaranteed unmasked entries)
- **Default BLOCK_SIZE=128** is good tradeoff

### 5. window_size Type

- **Use Python int** captured in closure as compile-time constant
- Changing it causes recompile - cache per distinct window_size
- Tensor doesn't help avoid recompiles

### 6. Llama Config

- **Llama configs (including 3.2) do NOT define sliding_window**
- **Mistral does** (config.sliding_window)
- Set SWA in attention backend, not from Llama config

## Complete Implementation

```python
import torch
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    and_masks,
)

# Base masks
def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def sliding_window(window_size: int, bidirectional: bool = False):
    W = int(window_size)
    if bidirectional:
        def _mask(b, h, q_idx, kv_idx):
            return (q_idx - kv_idx).abs() <= W
    else:
        def _mask(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= W)
    return _mask

def make_doc_mask(doc_id: torch.Tensor):
    def _doc(b, h, q_idx, kv_idx):
        return doc_id[b, q_idx] == doc_id[b, kv_idx]
    return _doc

# Compose: causal + sliding window (+ doc)
def build_combo_mask(window_size: int, doc_id=None, bidirectional=False):
    masks = [sliding_window(window_size, bidirectional=bidirectional)]
    if not bidirectional:
        masks.insert(0, causal)
    if doc_id is not None:
        masks.append(make_doc_mask(doc_id))
    return and_masks(*masks)

# Cache BlockMasks
_BLOCKMASK_CACHE = {}

def get_block_mask(B, H, S_q, S_kv, window_size, device, bidirectional=False, doc_id=None):
    key = (B, H, S_q, S_kv, int(window_size), bidirectional)
    if key not in _BLOCKMASK_CACHE:
        combo = build_combo_mask(window_size, doc_id=doc_id, bidirectional=bidirectional)
        _BLOCKMASK_CACHE[key] = create_block_mask(
            combo, B=B, H=H, Q_LEN=S_q, KV_LEN=S_kv, device=device
        )
    return _BLOCKMASK_CACHE[key]

# Usage
def attention_step(q, k, v, window_size, doc_id=None):
    B, H, S, D = q.shape
    bm = get_block_mask(B, H, S, S, window_size, q.device, bidirectional=False, doc_id=doc_id)
    return flex_attention(
        q, k, v,
        block_mask=bm,
        kernel_options={"ROWS_GUARANTEED_SAFE": True},
    )
```

## References

- FlexAttention API docs: https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html
- FlexAttention blog: https://docs.pytorch.org/blog/flexattention/
- Performance discussion: https://discuss.pytorch.org/t/training-with-flex-attention-is-extremely-slow-due-to-torch-compile-settings/222581
