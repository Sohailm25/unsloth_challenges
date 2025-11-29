# Oracle Research: Memory Efficient Backprop for Challenge E

## Description
GPT-5 Pro response on implementing memory-efficient backpropagation for LLM training with large vocabularies.

## Key Insights

### 1. Structure of autograd.Function
- Run forward in `no_grad`, calling a user-provided transform that returns per-element values (not reduced scalar)
- Save only small tensors/metadata
- In backward, replay forward per chunk with gradients enabled and call `torch.autograd.grad` on the transform's output

### 2. Handling Reductions
- Standardize on returning unreduced vector from transform
- For mean: maintain numerator and denominator (weights.sum()) per chunk
- In backward: scale grad_outputs with `weights/denom` so mean behaves correctly

### 3. Accumulating Weight Gradients
- In backward, recompute each chunk under grad
- Call `autograd.grad(outputs=chunk_values, inputs=(X_chunk, W, b), grad_outputs=scaled_upstream)`
- Add dW, db into running buffers

### 4. Recompute-in-Backward Pattern
- Forward: `with torch.no_grad()` - compute small outputs and aggregate
- Backward: `with torch.enable_grad()` - rebuild computation for each chunk
- Use grad_outputs that include upstream dY and reduction scaling

### 5. GRPO Support
- Transform computes per-token logprob for selected action
- Multiplies by per-token/group-wise advantages/weights
- Returns (values, weights_mask)
- The Function's reduction handles mean/sum across groups

## Implementation Pattern

```python
class MemoryEfficientLinear(Function):
    @staticmethod
    def forward(ctx, X, weight, bias, transform_fn, reduction, chunk_size, extras):
        # Forward in no_grad, chunked
        with torch.no_grad():
            for chunk in chunks:
                values, weights = transform_fn(chunk, linear_fn, **extras)
                accumulate(values, weights)

        ctx.save_for_backward(X, weight, ...)
        return output

    @staticmethod
    def backward(ctx, dY):
        # Backward with recomputation
        for chunk in chunks:
            with torch.enable_grad():
                values, weights = transform_fn(chunk, linear_fn, **extras)
                # Scale by upstream and reduction factor
                g_out = compute_grad_outputs(dY, weights, reduction)

                grads = torch.autograd.grad(values, [x_chunk, w, b], g_out)
                accumulate_grads(grads)

        return grad_X, grad_W, grad_b, ...
```

## Transform Function Pattern

```python
def ce_transform_unreduced(batch, linear_fn, labels, ignore_index=-100):
    logits = linear_fn(batch)  # [N_chunk, vocab]
    loss = F.cross_entropy(logits, labels, reduction="none", ignore_index=ignore_index)
    weights = (labels != ignore_index).to(loss.dtype)
    return loss, weights
```

## Key Points
1. NO float32 upcast - keep original dtype
2. NO hardcoded gradients - use autograd internally
3. Transform returns (values, weights) tuple
4. Support dynamic chunk sizes
5. Properly handle ignore_index in CE loss
