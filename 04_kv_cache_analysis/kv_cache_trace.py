from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUSINESS_APP_DIR = PROJECT_ROOT / "01_business_app"
REPORT_PATH = Path(__file__).with_name("kv_cache_report.md")
sys.path.insert(0, str(BUSINESS_APP_DIR))

from rag_pipeline import (  # noqa: E402
    DEFAULT_DOCUMENT,
    DEFAULT_QUESTION,
    build_prompt,
    get_generator,
    retrieve_context,
)


PROMPT_LENGTHS_TO_ESTIMATE = [512, 2048, 8192]


@dataclass
class LayerCacheInfo:
    layer_id: int
    k_shape: tuple[int, ...]
    v_shape: tuple[int, ...]
    dtype: str
    k_mb: float
    v_mb: float


def main():
    generator = get_generator()
    model_inputs = build_model_inputs(generator)

    with torch.inference_mode():
        outputs = generator.model(**model_inputs, use_cache=True)

    past_key_values = outputs.past_key_values
    layer_infos = inspect_kv_cache(past_key_values)
    seq_len = model_inputs["input_ids"].shape[-1]

    print_model_summary(generator, seq_len)
    print_layer_shapes(layer_infos)
    print_memory_summary(layer_infos)
    print_estimated_growth(generator)
    write_report(generator, seq_len, layer_infos)
    print(f"\nReport written to: {REPORT_PATH}")


def build_model_inputs(generator):
    retrieved_context = retrieve_context(DEFAULT_DOCUMENT, DEFAULT_QUESTION)
    prompt = build_prompt(retrieved_context, DEFAULT_QUESTION)
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


def inspect_kv_cache(past_key_values) -> list[LayerCacheInfo]:
    layer_infos: list[LayerCacheInfo] = []

    for layer_id, key_tensor, value_tensor in iter_kv_layers(past_key_values):
        layer_infos.append(
            LayerCacheInfo(
                layer_id=layer_id,
                k_shape=tuple(key_tensor.shape),
                v_shape=tuple(value_tensor.shape),
                dtype=str(key_tensor.dtype).replace("torch.", ""),
                k_mb=tensor_mb(key_tensor),
                v_mb=tensor_mb(value_tensor),
            )
        )

    return layer_infos


def iter_kv_layers(past_key_values):
    if hasattr(past_key_values, "layers"):
        for layer_id, layer in enumerate(past_key_values.layers):
            yield layer_id, layer.keys, layer.values
        return

    for layer_id, layer_cache in enumerate(past_key_values):
        key_tensor, value_tensor = layer_cache[:2]
        yield layer_id, key_tensor, value_tensor


def tensor_mb(tensor: torch.Tensor) -> float:
    return tensor.numel() * tensor.element_size() / 1024**2


def print_model_summary(generator, seq_len: int):
    config = generator.model.config
    print("[Model]")
    print(f"layers: {config.num_hidden_layers}")
    print(f"num_attention_heads: {config.num_attention_heads}")
    print(f"num_kv_heads: {config.num_key_value_heads}")
    print(f"hidden_size: {config.hidden_size}")
    print(f"head_dim: {config.hidden_size // config.num_attention_heads}")
    print(f"current seq_len: {seq_len}")


def print_layer_shapes(layer_infos: list[LayerCacheInfo]):
    print("\n[KV Cache Shapes]")
    for info in layer_infos:
        print(f"Layer {info.layer_id}:")
        print(f"  K shape = {list(info.k_shape)}")
        print(f"  V shape = {list(info.v_shape)}")
        print(f"  dtype = {info.dtype}")
        print(f"  K memory = {info.k_mb:.4f} MB")
        print(f"  V memory = {info.v_mb:.4f} MB")


def print_memory_summary(layer_infos: list[LayerCacheInfo]):
    total_mb = total_kv_cache_mb(layer_infos)
    print("\n[Current Request KV Cache]")
    print(f"layers: {len(layer_infos)}")
    print(f"total KV cache: {total_mb:.4f} MB")


def print_estimated_growth(generator):
    print("\n[Estimated KV Cache Growth]")
    for prompt_len in PROMPT_LENGTHS_TO_ESTIMATE:
        mb = estimate_kv_cache_mb(generator, prompt_len)
        print(f"Prompt len = {prompt_len}, KV Cache = {mb:.2f} MB")


def total_kv_cache_mb(layer_infos: list[LayerCacheInfo]) -> float:
    return sum(info.k_mb + info.v_mb for info in layer_infos)


def estimate_kv_cache_mb(generator, seq_len: int, batch_size: int = 1) -> float:
    config = generator.model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    dtype_bytes = torch.tensor([], dtype=generator.dtype).element_size()

    elements = batch_size * num_layers * 2 * num_kv_heads * seq_len * head_dim
    return elements * dtype_bytes / 1024**2


def write_report(generator, seq_len: int, layer_infos: list[LayerCacheInfo]):
    config = generator.model.config
    lines = [
        "# KV Cache Report",
        "",
        "## Model Config",
        "",
        f"- layers: `{config.num_hidden_layers}`",
        f"- num_attention_heads: `{config.num_attention_heads}`",
        f"- num_kv_heads: `{config.num_key_value_heads}`",
        f"- hidden_size: `{config.hidden_size}`",
        f"- head_dim: `{config.hidden_size // config.num_attention_heads}`",
        f"- current seq_len: `{seq_len}`",
        f"- dtype: `{layer_infos[0].dtype if layer_infos else 'unknown'}`",
        "",
        "## Per-Layer KV Shapes",
        "",
        "| Layer | K shape | V shape | K MB | V MB |",
        "| ---: | --- | --- | ---: | ---: |",
    ]

    for info in layer_infos:
        lines.append(
            f"| {info.layer_id} | `{list(info.k_shape)}` | `{list(info.v_shape)}` | "
            f"{info.k_mb:.4f} | {info.v_mb:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Current Request KV Memory",
            "",
            f"- total KV cache: `{total_kv_cache_mb(layer_infos):.4f} MB`",
            "",
            "## Estimated Growth",
            "",
            "| Prompt length | Estimated KV cache |",
            "| ---: | ---: |",
        ]
    )

    for prompt_len in PROMPT_LENGTHS_TO_ESTIMATE:
        lines.append(f"| {prompt_len} tokens | {estimate_kv_cache_mb(generator, prompt_len):.2f} MB |")

    lines.extend(
        [
            "",
            "## Formula",
            "",
            "```text",
            "KV cache bytes = batch_size * layers * 2 * num_kv_heads * seq_len * head_dim * dtype_bytes",
            "```",
            "",
            "The `2` accounts for K and V. For this model, each layer stores one K tensor and one V tensor.",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
