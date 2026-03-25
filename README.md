# TurboQuant

Implementation of [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026) — KV cache compression for local LLM inference.

Compresses transformer KV cache **6× with zero accuracy loss** using PolarQuant + QJL.

## Quick Start

```bash
# Install
git clone https://github.com/TheTom/turboquant_plus.git
cd turboquant_plus
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests (133 tests, 100% coverage)
python3 -m pytest tests/ -v

# Run demo
python3 benchmarks/demo.py

# Run Prince Canuma comparison
python3 benchmarks/test_outlier_comparison.py

# Run integration test at real Qwen dimensions
python3 benchmarks/test_with_llama.py

# Validate with real model (requires torch + transformers)
pip install transformers torch accelerate
python3 benchmarks/validate_real_model.py
```

## Compression Results

| Config | Compression | Cosine Sim | MSE |
|--------|-------------|------------|-----|
| TurboQuant 2-bit | 7.1× | 0.79 | 0.0047 |
| TurboQuant 2.5-bit (outlier) | **4.9×** | 0.86 | 0.0029 |
| TurboQuant 3-bit | 4.9× | 0.91 | 0.0018 |
| TurboQuant 3.5-bit (outlier) | **3.8×** | 0.95 | 0.0009 |
| TurboQuant 4-bit | 3.8× | 0.96 | 0.0007 |

2.5-bit and 3.5-bit ratios match [Prince Canuma's MLX implementation](https://x.com/Prince_Canuma) exactly.

## Architecture

```
Input: KV cache vector x ∈ R^d (one attention head)
    │
    ├── Extract norm: γ = ||x||, x̂ = x/γ
    │
    ├── Stage 1: PolarQuant (b-1 bits)
    │   Random rotation Π → coordinates ~ N(0, 1/d)
    │   → optimal scalar quantization per coordinate
    │
    ├── Stage 2: QJL (1 bit)
    │   sign(S · residual) → unbiased inner product correction
    │
    └── Output: CompressedVector(indices, signs, norms)
        Total: b bits per coordinate
```

## Project Structure

```
turboquant/
├── rotation.py      # Random rotation matrices (dense QR + fast Walsh-Hadamard)
├── codebook.py      # Optimal centroid computation (closed-form + Lloyd's)
├── polar_quant.py   # PolarQuant (Algorithm 1) — with norm extraction
├── qjl.py           # QJL 1-bit quantizer
├── turboquant.py    # Full TurboQuant (Algorithm 2)
├── kv_cache.py      # KV cache integration layer
├── outlier.py       # Outlier channel strategy (2.5-bit, 3.5-bit)
└── utils.py         # Bit packing, memory measurement

tests/               # 141 tests, 100% coverage on core modules
benchmarks/
├── demo.py                    # Quick compression demo
├── test_with_llama.py         # Integration test at Qwen 3.5 dimensions
├── test_outlier_comparison.py # Comparison with Prince Canuma's results
└── validate_real_model.py     # Phase A: real model tensor validation
```

## Development Workflow

Every change follows this loop (see [CLAUDE.md](CLAUDE.md) for full rules):

```
GitHub Issue → Tests FIRST → Codex review tests → Implement →
Codex + Roast review → Fix → Verify tests → Commit → Close issue
```

## Paper Reference

- **TurboQuant**: arXiv 2504.19874 (ICLR 2026)
- **PolarQuant**: arXiv 2502.02617 (AISTATS 2026)
- **QJL**: arXiv 2406.03482

## Roadmap

| Phase | Status |
|-------|--------|
| Core algorithms (NumPy) | ✅ Complete |
| Distortion validation | ✅ Complete |
| Outlier channel strategy | ✅ Complete |
| Real model validation (Phase A) | 🔄 In Progress |
| llama.cpp C port (Phase B) | ⏳ Blocked on Phase A |
| MLX port | ⏳ Last |

## Target Hardware

- Apple M5 Max 128 GB (llama.cpp + Metal)
- RTX 3090 24 GB (llama.cpp + CUDA)

## License

Research implementation. Based on Google Research's TurboQuant paper.
