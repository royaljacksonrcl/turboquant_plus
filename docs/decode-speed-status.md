# Decode Speed Status Report

## Current State (main branch)

turbo3 decode speed is 84% of q8_0 on all tested hardware. This is consistent and reproducible.

### Our Measurements (M5 Max 128GB, Qwen3.5-35B-A3B)

| Metric | turbo3 | q8_0 | Ratio |
|--------|--------|------|-------|
| Prefill (32-chunk) | 2777 tok/s | 2694 tok/s | 1.03x |
| Decode (8K ctx) | 65.8 tok/s | 78.3 tok/s | 0.84x |
| PPL (32-chunk) | 5.471 | 5.414 | +1.1% |
| KV cache size | 140 MiB | 340 MiB | 0.41x |

Prefill is at parity. Context scaling is flat (0.99x through 32K). Quality is within 1.1%. The only gap is **decode: 16% slower**.

### External Validation

| Tester | Hardware | Context | Decode ratio |
|--------|----------|---------|-------------|
| Mario (M1 Max 64GB) | M1 Max | 32K | 0.83x |
| Us (M5 Max 128GB) | M5 Max | 8K | 0.84x |
| Anon (M1 Max 64GB) | M1 Max | 42K | 0.36x (needs investigation) |

Mario's 0.83x matches ours closely. The anon tester's 0.36x is an outlier — likely different code version, no flash attention, or much deeper context.

## Root Cause

**Identified and verified:** The centroid table lookup in the flash attention dequant kernel.

Speed ceiling test (removing the LUT, returning constant values): turbo3 reaches **78.1 tok/s = exact q8_0 parity**. The entire 16% gap comes from 82 million data-dependent constant memory accesses per decode token.

q8_0 dequant: `int8_value * scale` (1 multiply per element)
turbo3 dequant: `centroid_table[3bit_index] * norm` (1 table lookup + 1 multiply per element)

The table lookup is the cost — it's data-dependent indexing that creates pipeline stalls on the GPU.

## What Was Tried (experiment/decode-speed-parity branch)

| Approach | Decode tok/s | vs baseline | Verdict |
|----------|-------------|-------------|---------|
| float32 LUT (main branch) | 65.8 | baseline | current |
| fp16 centroid LUT (vec only) | 67.4 | +2.4% | marginal win, offset by custom op overhead |
| Register float4 select | 63.1 | -4.1% | variable indexing equally bad |
| Inline switch (8 cases) | 57.3 | -12.9% | branches worse than LUT |
| Precomputed centroid*norm | 66.9 | +1.7% | marginal |
| fp16 LUT both paths | 59.2 | -10.0% | non-vec fp16 conversion overhead |
| No LUT (speed ceiling) | 78.1 | +18.7% | theoretical max |
| **Custom GGML_OP_TURBO_WHT + fp16 LUT** | **61.5** | **-6.5%** | **NET NEGATIVE — extra graph nodes hurt decode** |

**Key finding:** The custom GGML_OP_TURBO_WHT (which replaces the dense matmul with O(d log d) butterfly) adds graph overhead that makes decode WORSE. The matmul was never the decode bottleneck — it's negligible for single-token generation. The additional Metal kernel dispatches from the custom op cost more than they save.

## Conclusion

**The main branch (feature/turboquant-kv-cache) is the correct configuration for testers.** The experiment branch changes are a net negative.

The 16% decode gap is a fundamental cost of the 3-bit compression format — data-dependent table lookups on GPU are inherently slower than q8_0's simple integer scaling. Closing this gap requires either:

1. **Fused compressed attention** — compute Q·K directly on quantized indices, avoiding full dequant (complex, custom flash attention variant needed)
2. **Different block format** — store centroid values directly instead of indices (gives up compression)

Neither is trivial. The current 0.84x decode ratio is the best achievable within the existing block format and ggml op framework.

## Recommendation for Testers

- turbo3 decode at 0.83-0.84x of q8_0 is expected behavior
- Prefill is at parity (0.99-1.03x)
- The memory savings (2.4x smaller KV cache) enables longer context windows
- For latency-sensitive workloads where decode speed matters most, use q8_0
- For memory-constrained workloads (long context, large models), use turbo3
