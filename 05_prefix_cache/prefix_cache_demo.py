from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
import time

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUSINESS_APP_DIR = PROJECT_ROOT / "01_business_app"
REPORT_PATH = Path(__file__).with_name("reuse_report.md")
sys.path.insert(0, str(BUSINESS_APP_DIR))

from rag_pipeline import DEFAULT_DOCUMENT, get_generator  # noqa: E402


MAX_NEW_TOKENS = 16
DOCUMENT_REPEAT = 12
MIN_PREFIX_TOKENS = 128


@dataclass
class Request:
    request_id: str
    user_id: str
    session_id: str
    document_id: str
    document: str
    question: str

    @property
    def scope_key(self) -> str:
        return f"{self.user_id}:{self.session_id}:{self.document_id}"


@dataclass
class PrefixEntry:
    scope_key: str
    prefix_key: str
    prefix_tokens: int
    prefix_token_ids: list[int]
    past_key_values: object


@dataclass
class ReuseResult:
    user_id: str
    session_id: str
    document_id: str
    common_prefix_tokens: int
    common_prefix_ratio: float
    decision: str
    prefix_tokens: int
    suffix_tokens: int
    ttft_without_reuse: float
    ttft_with_reuse: float
    answer: str


class PrefixCache:
    def __init__(self):
        self.entries: dict[str, PrefixEntry] = {}

    def get(self, scope_key: str) -> PrefixEntry | None:
        return self.entries.get(scope_key)

    def put(
        self,
        scope_key: str,
        prefix_key: str,
        prefix_tokens: int,
        prefix_token_ids: list[int],
        past_key_values,
    ):
        self.entries[scope_key] = PrefixEntry(
            scope_key=scope_key,
            prefix_key=prefix_key,
            prefix_tokens=prefix_tokens,
            prefix_token_ids=prefix_token_ids,
            past_key_values=past_key_values,
        )


def main():
    generator = get_generator()
    cache = PrefixCache()

    demo_document = build_demo_document()
    request_001 = Request(
        request_id="request_001",
        user_id="user_A",
        session_id="session_doc_vllm",
        document_id="doc_vllm",
        document=demo_document,
        question="Which techniques does the document say vLLM uses?",
    )
    request_002 = Request(
        request_id="request_002",
        user_id="user_A",
        session_id="session_doc_vllm",
        document_id="doc_vllm",
        document=demo_document,
        question="Which techniques improve serving efficiency?",
    )

    print("[Prefix Cache Demo]")
    print("request_001 arrives first and creates a scoped prefix cache entry")
    print("request_002 arrives later and checks the existing scoped cache entry")

    miss_entry = handle_first_request(generator, cache, request_001)
    result = handle_incoming_request(
        generator,
        cache,
        request_002,
    )

    print("\n[Summary]")
    print(f"cached prefix tokens: {miss_entry.prefix_tokens}")
    print(f"request_002 suffix tokens: {result.suffix_tokens}")
    print(f"TTFT before reuse: {result.ttft_without_reuse:.4f}s")
    print(f"TTFT after reuse: {result.ttft_with_reuse:.4f}s")
    print(f"speedup: {result.ttft_without_reuse / result.ttft_with_reuse:.2f}x")
    print("\n[Answer]")
    print(result.answer)

    write_report(result)
    print(f"\nReport written to: {REPORT_PATH}")


def build_shared_prefix(document: str) -> str:
    return f"""You are a document question answering assistant.
Answer only from the document context.
Keep the answer short.

Document context:
{document.strip()}

Question:
"""


def build_demo_document() -> str:
    return "\n\n".join([DEFAULT_DOCUMENT.strip()] * DOCUMENT_REPEAT)


def build_suffix(question: str) -> str:
    return f"{question}\n\nAnswer:\n"


def analyze_incoming_request(generator, cache_entry: PrefixEntry | None, request: Request) -> dict:
    incoming_prefix = build_shared_prefix(request.document)
    incoming_prefix_ids = tokenize_text(generator, incoming_prefix)

    if cache_entry is None:
        return {
            "incoming_prefix_tokens": len(incoming_prefix_ids),
            "common_prefix_tokens": 0,
            "common_prefix_ratio": 0.0,
            "decision": "skip prefix cache",
            "reason": "no cache entry for this user/session/document scope",
            "shared_prefix": incoming_prefix,
        }

    common_prefix_tokens = count_common_prefix(cache_entry.prefix_token_ids, incoming_prefix_ids)
    common_prefix_ratio = common_prefix_tokens / min(len(cache_entry.prefix_token_ids), len(incoming_prefix_ids))

    if common_prefix_tokens < MIN_PREFIX_TOKENS:
        decision = "skip prefix cache"
        reason = f"common prefix below threshold {MIN_PREFIX_TOKENS}"
    elif common_prefix_tokens < cache_entry.prefix_tokens:
        decision = "skip prefix cache"
        reason = "incoming request does not fully match cached prefix"
    else:
        decision = "enable prefix cache"
        reason = "incoming request matches scoped cached prefix"

    return {
        "incoming_prefix_tokens": len(incoming_prefix_ids),
        "common_prefix_tokens": common_prefix_tokens,
        "common_prefix_ratio": common_prefix_ratio,
        "decision": decision,
        "reason": reason,
        "shared_prefix": incoming_prefix,
    }


def tokenize_text(generator, text: str) -> list[int]:
    return generator.tokenizer(text, add_special_tokens=False)["input_ids"]


def count_common_prefix(first_ids: list[int], second_ids: list[int]) -> int:
    count = 0
    for first_token, second_token in zip(first_ids, second_ids):
        if first_token != second_token:
            break
        count += 1
    return count


def print_analysis(analysis: dict):
    print("\n[Incoming Request Prefix Analysis]")
    print(f"incoming prefix tokens: {analysis['incoming_prefix_tokens']}")
    print(f"common prefix tokens: {analysis['common_prefix_tokens']}")
    print(f"common prefix ratio: {analysis['common_prefix_ratio'] * 100:.2f}%")
    print(f"decision: {analysis['decision']}")
    print(f"reason: {analysis['reason']}")


def handle_first_request(generator, cache: PrefixCache, request: Request) -> PrefixEntry:
    shared_prefix = build_shared_prefix(request.document)
    print(f"{request.request_id}: no previous cache lookup needed; create scoped prefix cache")
    return run_prefix_miss(generator, cache, request, shared_prefix)


def run_prefix_miss(generator, cache: PrefixCache, request: Request, shared_prefix: str) -> PrefixEntry:
    prefix_key = make_prefix_key(shared_prefix)
    existing_entry = cache.get(request.scope_key)
    if existing_entry is not None:
        print(f"{request.request_id}: scoped prefix already cached, reuse {existing_entry.prefix_tokens} tokens KV")
        return existing_entry

    print(f"{request.request_id}: prefix miss, run prefill")
    prefix_inputs = generator.tokenizer([shared_prefix], return_tensors="pt").to(generator.model.device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with torch.inference_mode():
        outputs = generator.model(**prefix_inputs, use_cache=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    prefix_tokens = prefix_inputs["input_ids"].shape[-1]
    prefix_token_ids = prefix_inputs["input_ids"][0].tolist()
    cache.put(request.scope_key, prefix_key, prefix_tokens, prefix_token_ids, outputs.past_key_values)
    return cache.get(request.scope_key)


def handle_incoming_request(
    generator,
    cache: PrefixCache,
    request: Request,
) -> ReuseResult:
    print(f"\n{request.request_id}: incoming request")
    print(f"{request.request_id}: lookup cache scope = {request.scope_key}")
    entry = cache.get(request.scope_key)
    analysis = analyze_incoming_request(generator, entry, request)
    print_analysis(analysis)

    if analysis["decision"] != "enable prefix cache" or entry is None:
        raise RuntimeError(f"{request.request_id}: prefix cache skipped: {analysis['reason']}")

    shared_prefix = analysis["shared_prefix"]
    suffix = build_suffix(request.question)

    ttft_without_reuse = measure_ttft_without_reuse(generator, shared_prefix, suffix)
    ttft_with_reuse, answer, suffix_tokens = generate_with_prefix_reuse(generator, entry, suffix)

    print(f"{request.request_id}: prefix hit, reuse {entry.prefix_tokens} tokens KV")
    print(f"{request.request_id}: skipped repeated prefix prefill")

    return ReuseResult(
        user_id=request.user_id,
        session_id=request.session_id,
        document_id=request.document_id,
        common_prefix_tokens=analysis["common_prefix_tokens"],
        common_prefix_ratio=analysis["common_prefix_ratio"],
        decision=analysis["decision"],
        prefix_tokens=entry.prefix_tokens,
        suffix_tokens=suffix_tokens,
        ttft_without_reuse=ttft_without_reuse,
        ttft_with_reuse=ttft_with_reuse,
        answer=answer,
    )


def measure_ttft_without_reuse(generator, shared_prefix: str, suffix: str) -> float:
    full_prompt = shared_prefix + suffix
    full_inputs = generator.tokenizer([full_prompt], return_tensors="pt").to(generator.model.device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        outputs = generator.model(**full_inputs, use_cache=True)
        _ = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.perf_counter() - start


def generate_with_prefix_reuse(generator, entry: PrefixEntry, suffix: str) -> tuple[float, str, int]:
    suffix_inputs = generator.tokenizer(
        [suffix],
        return_tensors="pt",
        add_special_tokens=False,
    ).to(generator.model.device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        outputs = generator.model(
            input_ids=suffix_inputs["input_ids"],
            past_key_values=entry.past_key_values,
            use_cache=True,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    ttft = time.perf_counter() - start
    token_id = int(next_token.item())
    generated_token_ids = [token_id]

    for _ in range(MAX_NEW_TOKENS - 1):
        if token_id == generator.tokenizer.eos_token_id:
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

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    answer = clean_answer(generator.tokenizer.decode(generated_token_ids, skip_special_tokens=True))
    suffix_tokens = suffix_inputs["input_ids"].shape[-1]
    return ttft, answer, suffix_tokens


def make_prefix_key(prefix: str) -> str:
    return hashlib.sha256(prefix.encode("utf-8")).hexdigest()


def clean_answer(answer: str) -> str:
    for marker in ["Human:", "Question:", "\n\n"]:
        answer = answer.split(marker, 1)[0]
    return answer.strip()


def write_report(result: ReuseResult):
    REPORT_PATH.write_text(
        f"""# Prefix Cache Reuse Report

## Setup

- Model: `models/Qwen2.5-1.5B-Instruct`
- Backend: Transformers local generation
- Shared prefix: instruction + document context + question header
- Document repeat count: `{DOCUMENT_REPEAT}`
- User/session/document: `{result.user_id}` / `{result.session_id}` / `{result.document_id}`
- Decision: `{result.decision}`
- Common prefix tokens: `{result.common_prefix_tokens}`
- Common prefix ratio: `{result.common_prefix_ratio * 100:.2f}%`
- Cached prefix tokens: `{result.prefix_tokens}`
- Request suffix tokens: `{result.suffix_tokens}`

## Result

| Metric | Value |
| --- | ---: |
| TTFT without prefix reuse | {result.ttft_without_reuse:.4f} s |
| TTFT with prefix reuse | {result.ttft_with_reuse:.4f} s |
| TTFT speedup | {result.ttft_without_reuse / result.ttft_with_reuse:.2f}x |

## Answer

```text
{result.answer}
```

## What This Demonstrates

- `request_001` arrives first, computes the shared prefix once, and stores a scoped KV cache entry.
- `request_002` arrives later, looks up the scoped cache by user/session/document, and then analyzes whether its incoming prefix matches the cached prefix.
- Because the incoming prefix fully matches the cached prefix, `request_002` reuses the cached prefix KV and only runs prefill for the question suffix.
- Prefix cache mainly reduces TTFT when many requests share the same system prompt or document prefix.
- This demo reuses one in-process Transformers KV cache object. Real serving systems usually store reusable prefix blocks in a managed KV cache pool.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
