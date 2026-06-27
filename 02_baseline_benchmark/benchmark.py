from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUSINESS_APP_DIR = PROJECT_ROOT / "01_business_app"
sys.path.insert(0, str(BUSINESS_APP_DIR))

from rag_pipeline import (  # noqa: E402
    DEFAULT_DOCUMENT,
    DEFAULT_QUESTION,
    build_prompt,
    get_generator,
    retrieve_context,
)


MAX_NEW_TOKENS = 128
REPORT_PATH = Path(__file__).with_name("baseline_report.md")


@dataclass
class BenchmarkResult:
    prompt_tokens: int
    output_tokens: int
    ttft_seconds: float
    tpot_seconds: float
    tokens_per_second: float
    total_latency_seconds: float
    peak_gpu_memory_gb: float
    answer: str


def main():
    result = run_baseline_benchmark()
    print_result(result)
    write_report(result)
    print(f"\nReport written to: {REPORT_PATH}")


def run_baseline_benchmark() -> BenchmarkResult:
    generator = get_generator()
    prompt = build_prompt(
        retrieve_context(DEFAULT_DOCUMENT, DEFAULT_QUESTION),
        DEFAULT_QUESTION,
    )
    model_inputs = build_model_inputs(generator, prompt)

    warmup(generator, prompt)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    generated_token_ids, first_token_time = generate_token_by_token(generator, model_inputs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    total_latency = end_time - start_time
    ttft = first_token_time - start_time
    output_tokens = len(generated_token_ids)
    decode_tokens = max(output_tokens - 1, 1)
    tpot = max(total_latency - ttft, 0.0) / decode_tokens
    tokens_per_second = output_tokens / total_latency if total_latency > 0 else 0.0
    peak_memory = get_peak_gpu_memory_gb()
    answer = generator.tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

    return BenchmarkResult(
        prompt_tokens=model_inputs["input_ids"].shape[-1],
        output_tokens=output_tokens,
        ttft_seconds=ttft,
        tpot_seconds=tpot,
        tokens_per_second=tokens_per_second,
        total_latency_seconds=total_latency,
        peak_gpu_memory_gb=peak_memory,
        answer=answer,
    )


def build_model_inputs(generator, prompt: str):
    messages = [
        {"role": "system", "content": "You are a helpful document question answering assistant."},
        {"role": "user", "content": prompt},
    ]
    chat_text = generator.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return generator.tokenizer([chat_text], return_tensors="pt").to(generator.model.device)


def warmup(generator, prompt: str):
    generator.generate(prompt, max_new_tokens=8)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def generate_token_by_token(generator, model_inputs) -> tuple[list[int], float]:
    generated_token_ids: list[int] = []
    eos_token_id = generator.tokenizer.eos_token_id

    with torch.inference_mode():
        outputs = generator.model(**model_inputs, use_cache=True)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    first_token_time = time.perf_counter()
    token_id = int(next_token.item())
    generated_token_ids.append(token_id)

    for _ in range(MAX_NEW_TOKENS - 1):
        if token_id == eos_token_id:
            break

        with torch.inference_mode():
            outputs = generator.model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        token_id = int(next_token.item())
        generated_token_ids.append(token_id)

    return generated_token_ids, first_token_time


def get_peak_gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def print_result(result: BenchmarkResult):
    print(f"Prompt length: {result.prompt_tokens} tokens")
    print(f"Output length: {result.output_tokens} tokens")
    print(f"TTFT: {result.ttft_seconds:.4f}s")
    print(f"TPOT: {result.tpot_seconds:.4f}s/token")
    print(f"Throughput: {result.tokens_per_second:.2f} tokens/s")
    print(f"Peak GPU Memory: {result.peak_gpu_memory_gb:.2f} GB")
    print(f"Total latency: {result.total_latency_seconds:.4f}s")
    print("\n[Answer]")
    print(result.answer)


def write_report(result: BenchmarkResult):
    REPORT_PATH.write_text(
        f"""# Baseline Benchmark Report

## Setup

- Business path: `01_business_app`
- Model: `models/Qwen2.5-1.5B-Instruct`
- Backend: Transformers local generation
- Max new tokens: `{MAX_NEW_TOKENS}`

## Metrics

| Metric | Value |
| --- | ---: |
| Prompt length | {result.prompt_tokens} tokens |
| Output length | {result.output_tokens} tokens |
| TTFT | {result.ttft_seconds:.4f} s |
| TPOT | {result.tpot_seconds:.4f} s/token |
| Throughput | {result.tokens_per_second:.2f} tokens/s |
| Peak GPU Memory | {result.peak_gpu_memory_gb:.2f} GB |
| Total latency | {result.total_latency_seconds:.4f} s |

## Answer

```text
{result.answer}
```

## Notes

- TTFT is measured as time from entering model execution to the first decoded token.
- TPOT is measured from the remaining decode time divided by the remaining output tokens.
- This is the unoptimized Transformers baseline for later scheduler, KV cache, prefix cache, and vLLM comparisons.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
