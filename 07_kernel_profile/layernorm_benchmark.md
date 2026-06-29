# LayerNorm Kernel Benchmark

## Setup

- Device: `cuda`
- Rows: `512`
- Warmup iterations: `10`
- Benchmark iterations: `50`
- Baseline: `torch.nn.functional.layer_norm`
- Custom path: small explicit PyTorch implementation of LayerNorm math

## Results

| Hidden size | Max error | Torch time | Custom time | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 0.000002 | 0.0104 ms | 0.0746 ms | 0.14x |
| 1024 | 0.000001 | 0.0106 ms | 0.0830 ms | 0.13x |
| 2048 | 0.000002 | 0.0122 ms | 0.0969 ms | 0.13x |
| 4096 | 0.000002 | 0.0173 ms | 0.1243 ms | 0.14x |

## What This Shows

- Correctness comes first: the custom math is compared with PyTorch by max error.
- Benchmarking needs warmup and synchronization, especially on CUDA.
- This file is a minimal kernel-development loop. Replacing `naive_layernorm` with Triton or CUDA is the next step.
