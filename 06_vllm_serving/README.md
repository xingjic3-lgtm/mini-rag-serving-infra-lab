# Phase 06: vLLM Serving

This phase compares the local Transformers baseline with a vLLM OpenAI-compatible server.

Check the current environment:

```bash
python 06_vllm_serving/vllm_env_check.py
```

Run the server in one terminal:

```bash
bash 06_vllm_serving/vllm_server.sh
```

Run the benchmark client in another terminal:

```bash
python 06_vllm_serving/vllm_client.py
```

The client writes `06_vllm_serving/vllm_benchmark.md`.

If no server is reachable, the client writes a blocked report instead of fake metrics.
