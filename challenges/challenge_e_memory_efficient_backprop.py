# ABOUTME: Explains memory efficient backprop challenge and starter code.
# ABOUTME: Includes autograd function skeleton.

r"""
<a name="MATH"></a>
## E) Memory Efficient Backprop [Difficulty: Medium to Hard] [Max points: 10]

In LLMs, the last layer is a projection matrix to calculate the probabilities of the next token, ie $\sigma(XW)$. However, if the vocabulary size is very large, say 128K, then the materialization of the logits causes VRAM spikes.

For example, if the `bsz = 4, qlen = 4096, hd = 4096, vocab = 128K`, then the memory usage for the logits in bfloat16 would be 4GB. In the worst case, we might even need to upcast logits to float32, so 8GB is needed.

In Unsloth, we utilize [Apple's Cut Cross Entropy Loss](https://machinelearning.apple.com/research/cut-your-losses) to reduce VRAM usage, by allowing a Triton kernel to create the logits on the fly to calculate the cross entropy loss. But this does not generalize well to other functions.

Our goal is to generalize this ultimately, but directly creating logits on the fly will be hard. Instead, let's take a slightly less complex approach. Let's first review some stuff. We first notice that during the normal case after forming the intermediate logits for 2 batches, we then do a gather function to aggregate the intermediate results into a single column:
$$
\begin{align}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \times W &= \begin{bmatrix} x_1 W \\ x_2 W \end{bmatrix} \\
f \bigg( \begin{bmatrix} x_1 W \\ x_2 W \end{bmatrix} \bigg) &= \begin{pmatrix} y_1 \\ y_2 \end{pmatrix}
\end{align}
$$

So, if we can somehow skip the materialization of the intermediate logits, and just output the output of `f`, we can save a lot of VRAM!

Notice during backpropagation we can use the chain rule:
$$
\begin{align}
\frac{dL}{dX} &= \frac{dL}{dy} \frac{dy}{dX} ; \frac{dL}{dW} = \frac{dL}{dy} \frac{dy}{dW} \\
\frac{dL}{dy} &= \text{Downstream from backprop} \\
\frac{dy}{dX} &= W^T \\
\frac{dy}{dW} &= X^T \\
\frac{dL}{dX} &= \frac{dL}{dy} W^T \\
\frac{dL}{dW} &= X^T \frac{dL}{dy} \\
\end{align}
$$

If we simply compute the intermediate tensors on the fly via batches, say we do batch 1, then batch 2, we can reduce VRAM usage from 4GB to 2GB!

$$
\begin{align}
\frac{dL}{dX} &= \begin{bmatrix} \frac{dL_1}{dy_1} W^T \\ \frac{dL_2}{dy_2} W^T \end{bmatrix} \\
\frac{dL}{dW} &= \bigg( X_1^T \frac{dL_1}{dy_1} + X_2^T  \frac{dL_2}{dy_2} \bigg)
\end{align}
$$

1. Your goal is to write a `torch.autograd.Function` with a `forward` and `backward` pass showcasing this memory efficient implementation.

2. You must NOT hard code the derivatives - move the transformation function from the logits / intermeditate tensors to a smaller tensor as a separate function which can allow `autograd` to pass through it.

3. As a hint, look at `torch.checkpoint` at https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py. Also, don't forget about the upstream gradients! We need to multiply them to the current gradients!

4. Make the Cross Entropy Loss work. You must show other functions working as well.
"""

def transformation_function(batch, linear, labels):
    x = linear(batch).float() # Up projection to large space
    from torch.nn import CrossEntropyLoss
    down_projection_function = CrossEntropyLoss(reduction = "mean")
    # Down projection to small space
    loss = down_projection_function(x.view(-1, x.shape[-1]), labels.view(-1))
    return loss

class MemoryEfficientLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, linear, labels, forward_function):
        outputs = []
        # EDIT THIS FUNCTION
        output = forward_function(X, linear, labels)
        ctx.save_for_backward(X)
        return output

    @staticmethod
    def backward(ctx, dY):
        X = ctx.saved_tensors
        # EDIT THIS FUNCTION
        return X, None, None, None

"""To test your implementation, it should not OOM for large inputs. Also, check the gradient is actually equivalent via `torch.allclose` in the normal approach.

## Marking Criteria for E) Max points = 10
```python
if attemped_E:
    E_score = 0
    if VRAM_50_percent_reduction: E_score += 2
    if remove_float32_upcast: E_score = 0
    if show_ce_loss_works: E_score += 1
    if show_other_functions_work: E_score += 1
    if hardcoded_gradients: E_score = 0
    if allows_dynamic_chunk_sizes: E_score += 1
    if llama_1B_training_loss_matches: E_score += 1
    else: E_score = 0
    if GRPO_memory_efficient_linear_works: E_score += 4
    final_score += E_score
else:
    final_score += 0
```

---
---
---
"""
