from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.profiler import ProfilerActivity, profile, record_function


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUSINESS_APP_DIR = PROJECT_ROOT / "01_business_app"
REPORT_PATH = Path(__file__).with_name("profile_report.md")
TRACE_DIR = Path(__file__).with_name("nsight_trace")
sys.path.insert(0, str(BUSINESS_APP_DIR))

from rag_pipeline import DEFAULT_DOCUMENT, DEFAULT_QUESTION, build_prompt, get_generator, retrieve_context  # noqa: E402


DECODE_STEPS = 8


def main():
    generator = get_generator()
    model_inputs = build_model_inputs(generator)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.synchronize()

    with profile(activities=activities, record_shapes=True, with_stack=False) as prof:
        with record_function("prefill"):
            with torch.inference_mode():
                outputs = generator.model(**model_inputs, use_cache=True)
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                past_key_values = outputs.past_key_values

        with record_function("decode"):
            for _ in range(DECODE_STEPS):
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

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    chrome_trace_path = TRACE_DIR / "torch_profiler_trace.json"
    prof.export_chrome_trace(str(chrome_trace_path))

    table = prof.key_averages().table(
        sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
        row_limit=12,
    )
    print(table)

    has_cuda_columns = "CUDA" in table
    write_report(table, chrome_trace_path, torch.cuda.is_available(), has_cuda_columns)
    print(f"\nReport written to: {REPORT_PATH}")
    print(f"Trace written to: {chrome_trace_path}")


def build_model_inputs(generator):
    prompt = build_prompt(retrieve_context(DEFAULT_DOCUMENT, DEFAULT_QUESTION), DEFAULT_QUESTION)
    messages = [
        {"role": "system", "content": "You are a helpful document question answering assistant."},
        {"role": "user", "content": prompt},
    ]
    chat_text = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return generator.tokenizer([chat_text], return_tensors="pt").to(generator.model.device)


def write_report(table: str, chrome_trace_path: Path, has_cuda: bool, has_cuda_columns: bool):
    if has_cuda_columns:
        device_note = "CUDA profiler activities are included."
    elif has_cuda:
        device_note = "CUDA is available, but CUPTI did not provide CUDA activity timing in this run."
    else:
        device_note = "CUDA is not available; this is a CPU-only trace."
    REPORT_PATH.write_text(
        f"""# Kernel Profile Report

## Profile Command

```bash
python 07_kernel_profile/profile_torch_decode.py
```

Optional Nsight command:

```bash
nsys profile -o 07_kernel_profile/nsight_trace/vllm_or_transformers_trace python 07_kernel_profile/profile_torch_decode.py
```

## Trace Artifact

- Torch profiler Chrome trace: `{chrome_trace_path.as_posix()}`
- Device note: {device_note}

## Top Operators

```text
{table}
```

## Bottleneck Judgment

- Prefill processes the full prompt, so attention and matrix multiplication work scale with prompt length.
- Decode processes one new token at a time but repeatedly reads historical KV cache.
- If CUDA time is dominated by matrix multiplication operators, the current bottleneck is compute-heavy model layers.
- If CUDA time is dominated by attention or cache-related kernels, the current bottleneck is KV-cache read bandwidth and attention scheduling.
- If CPU time is large while CUDA time is small, the current bottleneck is request scheduling or Python/runtime overhead.

## Next Optimization Direction

- Compare this trace with vLLM serving under concurrency.
- Check whether decode has many small kernels, which usually points to batching or launch-overhead problems.
- Use Nsight Systems when you need the CPU scheduling gap between requests and GPU kernels.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
