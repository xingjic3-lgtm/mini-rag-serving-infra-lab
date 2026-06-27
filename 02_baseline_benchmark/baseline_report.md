# Baseline Benchmark Report

## Setup

- Business path: `01_business_app`
- Model: `models/Qwen2.5-1.5B-Instruct`
- Backend: Transformers local generation
- Max new tokens: `128`

## Metrics

| Metric | Value |
| --- | ---: |
| Prompt length | 171 tokens |
| Output length | 21 tokens |
| TTFT | 0.0301 s |
| TPOT | 0.0261 s/token |
| Throughput | 38.04 tokens/s |
| Peak GPU Memory | 2.96 GB |
| Total latency | 0.5520 s |

## Answer

```text
vLLM uses techniques such as paged KV cache management and continuous batching to improve serving efficiency.
```

## Notes

- TTFT is measured as time from entering model execution to the first decoded token.
- TPOT is measured from the remaining decode time divided by the remaining output tokens.
- This is the unoptimized Transformers baseline for later scheduler, KV cache, prefix cache, and vLLM comparisons.
