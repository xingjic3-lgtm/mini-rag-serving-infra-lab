# vLLM Serving Benchmark

## Status

`blocked`

## What Was Checked

- vLLM endpoint: `http://127.0.0.1:8000/v1`
- Local Python module `vllm`: `missing`
- Platform: `Windows`
- Model path: `models/Qwen2.5-1.5B-Instruct`

## Blocking Reason

```text
Cannot connect to vLLM server at http://127.0.0.1:8000/v1. platform=Windows, vllm_module=missing. On native Windows, official vLLM GPU serving is not supported; run vLLM in WSL/Linux or on a Linux GPU host, then rerun this client.
```

## Why This Is Not A Real vLLM Benchmark Yet

No vLLM server was reachable, so the script did not run the Transformers baseline or write fake vLLM numbers.

## How To Finish Phase 06

Run vLLM on a supported Linux/WSL GPU environment:

```bash
cd /mnt/d/Project/cudaproject/project
bash 06_vllm_serving/vllm_server.sh
```

Then rerun:

```bash
python 06_vllm_serving/vllm_client.py
```

When the server is reachable, this file will be replaced with a real comparison table:

```text
Engine          TTFT      TPOT      tokens/s      Peak Memory      Max Concurrency
transformers    ...
vLLM            ...
```

## Current Conclusion

Phase 06 is wired correctly as a vLLM client, but this Windows-native environment cannot currently execute the official vLLM server.
