# Prefix Cache Reuse Report

## Setup

- Model: `models/Qwen2.5-1.5B-Instruct`
- Backend: Transformers local generation
- Shared prefix: instruction + document context + question header
- Document repeat count: `12`
- User/session/document: `user_A` / `session_doc_vllm` / `doc_vllm`
- Decision: `enable prefix cache`
- Common prefix tokens: `1381`
- Common prefix ratio: `100.00%`
- Cached prefix tokens: `1381`
- Request suffix tokens: `8`

## Result

| Metric | Value |
| --- | ---: |
| TTFT without prefix reuse | 0.2308 s |
| TTFT with prefix reuse | 0.1503 s |
| TTFT speedup | 1.54x |

## Answer

```text
Paged KV cache management and continuous batching improve serving efficiency.
```

## What This Demonstrates

- `request_001` arrives first, computes the shared prefix once, and stores a scoped KV cache entry.
- `request_002` arrives later, looks up the scoped cache by user/session/document, and then analyzes whether its incoming prefix matches the cached prefix.
- Because the incoming prefix fully matches the cached prefix, `request_002` reuses the cached prefix KV and only runs prefill for the question suffix.
- Prefix cache mainly reduces TTFT when many requests share the same system prompt or document prefix.
- This demo reuses one in-process Transformers KV cache object. Real serving systems usually store reusable prefix blocks in a managed KV cache pool.
