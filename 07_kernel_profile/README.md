# Phase 07: Kernel Profile

Generate a local prefill/decode profile:

```bash
python 07_kernel_profile/profile_torch_decode.py
```

Run the LayerNorm correctness and benchmark loop:

```bash
python 07_kernel_profile/layernorm_kernel.py
```

Artifacts are written to:

- `07_kernel_profile/profile_report.md`
- `07_kernel_profile/nsight_trace/torch_profiler_trace.json`
- `07_kernel_profile/layernorm_benchmark.md`
