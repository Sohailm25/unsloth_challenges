🧿 oracle 0.5.0 — Session t4-flash-attn-plan
Prompt:
Need a pragmatic path to 'verify' flash-attn on T4/SM75 for unified attention work. Constraints: flash-attn wheels for torch 2.4/2.5 cu121 cp310/cp311 not found; binary install failed earlier. Building from source on T4 timed out. We already gate flash-attn off for <SM80. Goal: either produce a minimal verification (even CPU/fake) that flash-attn path is reachable, or justify documenting unavailability for SM75. Please suggest: (1) any available prebuilt wheels or config for T4 (maybe torch 2.3/2.2 + cu118?) that still lets us prove flash-attn works; (2) if impossible, the best doc wording + an automated test/guard we can add to enforce fallback on SM75; (3) an ultra-fast build recipe that could succeed on T4 under tight time (flags, subsets).

Answer (gpt-5-pro):
- FA2 is Ampere+ only; T4/SM75 unsupported per upstream. Wheels/builds won’t make FA2 run there.
- Validation options: (A) stub FA2 forward in tests and force capabilities to show dispatch hits flash path without GPU; (B) run one real FA2 smoke on any Ampere/Ada/Hopper; (C) if needed to show kernels on T4, use FlashAttention 1.x separately (not the unified FA2 path).
- Documentation text: state FA2 unavailable on SM75 (T4/RTX20xx); unified attention disables FA on compute capability <8.0 and falls back to xformers/SDPA. Users can force sdpa/xformers via UNSLOTH_ATTENTION; forcing flash on SM75 is ignored.
- Guard tests to add: pick_backend_sm75_falls_back, pick_backend_ampere_picks_flash.
- Fast build ideas: build FA1 with TORCH_CUDA_ARCH_LIST=7.5 on a faster host and pip install wheel on T4; or try flash-attn<2 build locally with MAX_JOBS=1, though FA2 on SM75 still won’t execute.
