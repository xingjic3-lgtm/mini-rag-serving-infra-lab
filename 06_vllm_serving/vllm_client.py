from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import json
import platform
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import build_opener
from urllib.request import Request as HttpRequest
from urllib.request import ProxyHandler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUSINESS_APP_DIR = PROJECT_ROOT / "01_business_app"
BASELINE_DIR = PROJECT_ROOT / "02_baseline_benchmark"
REPORT_PATH = Path(__file__).with_name("vllm_benchmark.md")
sys.path.insert(0, str(BUSINESS_APP_DIR))
sys.path.insert(0, str(BASELINE_DIR))

from benchmark import run_baseline_benchmark  # noqa: E402
from rag_pipeline import DEFAULT_DOCUMENT, DEFAULT_QUESTION, build_prompt, retrieve_context  # noqa: E402


API_BASE_URL = "http://127.0.0.1:8000/v1"
API_KEY = "local-token"
MODEL_NAME = "models/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 128
REQUEST_TIMEOUT_SECONDS = 5
NO_PROXY_OPENER = build_opener(ProxyHandler({}))


@dataclass
class ServingResult:
    engine: str
    prompt_tokens: int
    output_tokens: int
    ttft_seconds: float
    tpot_seconds: float
    tokens_per_second: float
    total_latency_seconds: float
    peak_gpu_memory_gb: float | None
    max_concurrency: int
    answer: str


@dataclass
class ServerStatus:
    available: bool
    reason: str
    served_model: str | None = None


def main():
    server_status = probe_vllm_server()
    if not server_status.available:
        print("[vLLM Serving]")
        print(server_status.reason)
        write_blocked_report(server_status)
        print(f"\nReport written to: {REPORT_PATH}")
        return

    print("[Transformers Baseline]")
    baseline = run_transformers_baseline()
    print_result(baseline)

    print("\n[vLLM Serving]")
    vllm = run_vllm_benchmark(server_status.served_model or MODEL_NAME)
    print_result(vllm)

    write_report([baseline, vllm])
    print(f"\nReport written to: {REPORT_PATH}")


def run_transformers_baseline() -> ServingResult:
    result = run_baseline_benchmark()
    return ServingResult(
        engine="transformers",
        prompt_tokens=result.prompt_tokens,
        output_tokens=result.output_tokens,
        ttft_seconds=result.ttft_seconds,
        tpot_seconds=result.tpot_seconds,
        tokens_per_second=result.tokens_per_second,
        total_latency_seconds=result.total_latency_seconds,
        peak_gpu_memory_gb=result.peak_gpu_memory_gb,
        max_concurrency=1,
        answer=result.answer,
    )


def run_vllm_benchmark(model_name: str) -> ServingResult:
    prompt = build_prompt(retrieve_context(DEFAULT_DOCUMENT, DEFAULT_QUESTION), DEFAULT_QUESTION)
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful document question answering assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0,
        "stream": True,
    }

    start = time.perf_counter()
    first_token_time: float | None = None
    generated_chunks: list[str] = []
    chunk_count = 0

    for chunk in stream_chat_completion(payload):
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content")
        if not content:
            continue
        if first_token_time is None:
            first_token_time = time.perf_counter()
        generated_chunks.append(content)
        chunk_count += 1

    end = time.perf_counter()
    if first_token_time is None:
        raise RuntimeError("vLLM returned no streamed content. Check the server logs.")

    answer = "".join(generated_chunks).strip()
    total_latency = end - start
    ttft = first_token_time - start
    output_tokens = max(chunk_count, 1)
    decode_tokens = max(output_tokens - 1, 1)
    tpot = max(total_latency - ttft, 0.0) / decode_tokens
    tokens_per_second = output_tokens / total_latency if total_latency > 0 else 0.0

    return ServingResult(
        engine="vLLM",
        prompt_tokens=0,
        output_tokens=output_tokens,
        ttft_seconds=ttft,
        tpot_seconds=tpot,
        tokens_per_second=tokens_per_second,
        total_latency_seconds=total_latency,
        peak_gpu_memory_gb=None,
        max_concurrency=1,
        answer=answer,
    )


def probe_vllm_server() -> ServerStatus:
    module_found = importlib.util.find_spec("vllm") is not None
    system_name = platform.system()
    platform_note = f"platform={system_name}, vllm_module={'found' if module_found else 'missing'}"
    request = HttpRequest(
        f"{API_BASE_URL}/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
        method="GET",
    )

    try:
        with NO_PROXY_OPENER.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        return ServerStatus(False, f"vLLM server responded with HTTP {error.code}: {detail}")
    except URLError:
        reason = (
            f"Cannot connect to vLLM server at {API_BASE_URL}. "
            f"{platform_note}. On native Windows, official vLLM GPU serving is not supported; "
            "run vLLM in WSL/Linux or on a Linux GPU host, then rerun this client."
        )
        return ServerStatus(False, reason)

    models = payload.get("data", [])
    served_model = models[0].get("id") if models else MODEL_NAME
    return ServerStatus(True, "vLLM server is reachable", served_model=served_model)


def stream_chat_completion(payload: dict):
    request = HttpRequest(
        f"{API_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with NO_PROXY_OPENER.open(request, timeout=300) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break
                yield json.loads(data)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM HTTP error {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(
            "Cannot connect to vLLM server. Start it first with "
            "`bash 06_vllm_serving/vllm_server.sh`."
        ) from error


def print_result(result: ServingResult):
    peak = "n/a" if result.peak_gpu_memory_gb is None else f"{result.peak_gpu_memory_gb:.2f} GB"
    print(f"Engine: {result.engine}")
    print(f"Output length: {result.output_tokens} streamed chunks/tokens")
    print(f"TTFT: {result.ttft_seconds:.4f}s")
    print(f"TPOT: {result.tpot_seconds:.4f}s/token")
    print(f"Throughput: {result.tokens_per_second:.2f} tokens/s")
    print(f"Peak GPU Memory: {peak}")
    print(f"Total latency: {result.total_latency_seconds:.4f}s")
    print("[Answer]")
    print(result.answer)


def write_report(results: list[ServingResult]):
    lines = [
        "# vLLM Serving Benchmark",
        "",
        "## Setup",
        "",
        "- Business path: `01_business_app`",
        "- Baseline: local Transformers token-by-token generation",
        "- vLLM endpoint: `http://127.0.0.1:8000/v1/chat/completions`",
        "- Model: `models/Qwen2.5-1.5B-Instruct`",
        "- Max new tokens: `128`",
        "",
        "## Comparison",
        "",
        "| Engine | TTFT | TPOT | tokens/s | Peak Memory | Max Concurrency |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in results:
        peak = "n/a" if result.peak_gpu_memory_gb is None else f"{result.peak_gpu_memory_gb:.2f} GB"
        lines.append(
            f"| {result.engine} | {result.ttft_seconds:.4f}s | "
            f"{result.tpot_seconds:.4f}s/token | {result.tokens_per_second:.2f} | "
            f"{peak} | {result.max_concurrency} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- vLLM exposes an OpenAI-compatible HTTP API, so the business client only changes the endpoint.",
            "- This script measures TTFT from HTTP request start to the first streamed content chunk.",
            "- vLLM server-side peak memory is not available through the OpenAI-compatible response, so this report marks it as `n/a` unless you collect it separately with `nvidia-smi` or Nsight.",
            "- The most important serving change is moving from single-process local generation to a serving engine that can batch multiple requests.",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_blocked_report(status: ServerStatus):
    vllm_module = "found" if importlib.util.find_spec("vllm") else "missing"
    REPORT_PATH.write_text(
        f"""# vLLM Serving Benchmark

## Status

`blocked`

## What Was Checked

- vLLM endpoint: `{API_BASE_URL}`
- Local Python module `vllm`: `{vllm_module}`
- Platform: `{platform.system()}`
- Model path: `{MODEL_NAME}`

## Blocking Reason

```text
{status.reason}
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
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
