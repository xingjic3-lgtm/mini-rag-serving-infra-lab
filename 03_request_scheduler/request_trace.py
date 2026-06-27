from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUSINESS_APP_DIR = PROJECT_ROOT / "01_business_app"
REPORT_PATH = PROJECT_ROOT / "reports" / "request_lifecycle.md"
sys.path.insert(0, str(BUSINESS_APP_DIR))

from rag_pipeline import (  # noqa: E402
    DEFAULT_DOCUMENT,
    DEFAULT_QUESTION,
    build_prompt,
    get_generator,
    retrieve_context,
)


REQUEST_ID = "request_001"
MAX_DECODE_STEPS = 32


@dataclass
class TraceEvent:
    request_id: str
    stage: str
    elapsed_seconds: float
    gpu_memory_gb: float
    detail: str


def main():
    events, answer = trace_request_lifecycle()
    write_report(events, answer)
    print(f"\nReport written to: {REPORT_PATH}")


def trace_request_lifecycle() -> tuple[list[TraceEvent], str]:
    events: list[TraceEvent] = []
    generator = get_generator()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    trace_start = time.perf_counter()

    def record(stage: str, detail: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        event = TraceEvent(
            request_id=REQUEST_ID,
            stage=stage,
            elapsed_seconds=time.perf_counter() - trace_start,
            gpu_memory_gb=current_gpu_memory_gb(),
            detail=detail,
        )
        events.append(event)
        print(format_event(event))

    record("enter queue", "raw request received")

    record("preprocess start", "retrieve context, build prompt, tokenize")
    retrieved_context = retrieve_context(DEFAULT_DOCUMENT, DEFAULT_QUESTION)
    prompt = build_prompt(retrieved_context, DEFAULT_QUESTION)
    model_inputs = build_model_inputs(generator, prompt)
    prompt_len = model_inputs["input_ids"].shape[-1]
    record("preprocess end", f"prompt_len={prompt_len}")

    record("prefill start", "run full prompt through model and create KV cache")

    with torch.inference_mode():
        outputs = generator.model(**model_inputs, use_cache=True)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values

    token_id = int(next_token.item())
    generated_token_ids = [token_id]
    token_text = decode_token(generator, token_id)
    record("prefill end", f"KV cache ready, first_token={token_text!r}, token_id={token_id}")

    eos_token_id = generator.tokenizer.eos_token_id
    for step in range(2, MAX_DECODE_STEPS + 1):
        if token_id == eos_token_id:
            record("decode stop", "eos token reached")
            break

        with torch.inference_mode():
            outputs = generator.model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values

        token_id = int(next_token.item())
        generated_token_ids.append(token_id)
        token_text = decode_token(generator, token_id)
        record(f"decode step={step}", f"new_token={token_text!r}, token_id={token_id}")

    answer = generator.tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()
    record("finished", f"output_tokens={len(generated_token_ids)}, total_latency={events[-1].elapsed_seconds:.4f}s")
    return events, answer


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


def decode_token(generator, token_id: int) -> str:
    text = generator.tokenizer.decode([token_id], skip_special_tokens=True)
    return text.replace("\n", "\\n")


def current_gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024**3


def format_event(event: TraceEvent) -> str:
    return (
        f"[{event.request_id}] {event.stage}, "
        f"t={event.elapsed_seconds:.4f}s, "
        f"gpu_mem={event.gpu_memory_gb:.2f}GB, "
        f"{event.detail}"
    )


def write_report(events: list[TraceEvent], answer: str):
    lines = [
        "# Request Lifecycle Trace",
        "",
        "## Request",
        "",
        f"- request_id: `{REQUEST_ID}`",
        f"- max_decode_steps: `{MAX_DECODE_STEPS}`",
        "- backend: Transformers local generation",
        "- model: `models/Qwen2.5-1.5B-Instruct`",
        "",
        "## Timeline",
        "",
        "| Stage | Elapsed | GPU Memory | Detail |",
        "| --- | ---: | ---: | --- |",
    ]

    for event in events:
        lines.append(
            f"| `{event.stage}` | {event.elapsed_seconds:.4f}s | "
            f"{event.gpu_memory_gb:.2f} GB | {event.detail} |"
        )

    lines.extend(
        [
            "",
            "## Final Answer",
            "",
            "```text",
            answer,
            "```",
            "",
            "## What This Shows",
            "",
            "- `enter queue` is where the business request first enters the serving system.",
            "- `prefill start` is where the full prompt is processed and KV cache is created.",
            "- each `decode step` consumes the previous token plus KV cache and emits one new token.",
            "- `finished` marks the request leaving the serving path.",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
