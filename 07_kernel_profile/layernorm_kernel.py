from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import torch


REPORT_PATH = Path(__file__).with_name("layernorm_benchmark.md")
HIDDEN_SIZES = [512, 1024, 2048, 4096]
ROWS = 512
WARMUP_ITERS = 10
BENCH_ITERS = 50


@dataclass
class BenchResult:
    hidden_size: int
    max_error: float
    torch_ms: float
    custom_ms: float
    speedup: float


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = [benchmark_hidden_size(hidden_size, device) for hidden_size in HIDDEN_SIZES]

    print("[LayerNorm Benchmark]")
    for result in results:
        print(
            f"hidden={result.hidden_size}, max error={result.max_error:.6f}, "
            f"torch={result.torch_ms:.4f} ms, custom={result.custom_ms:.4f} ms, "
            f"speedup={result.speedup:.2f}x"
        )

    write_report(results, device)
    print(f"\nReport written to: {REPORT_PATH}")


def benchmark_hidden_size(hidden_size: int, device: str) -> BenchResult:
    torch.manual_seed(0)
    x = torch.randn(ROWS, hidden_size, device=device, dtype=torch.float32)
    weight = torch.randn(hidden_size, device=device, dtype=torch.float32)
    bias = torch.randn(hidden_size, device=device, dtype=torch.float32)

    expected = torch.nn.functional.layer_norm(x, (hidden_size,), weight, bias)
    actual = naive_layernorm(x, weight, bias)
    max_error = (expected - actual).abs().max().item()

    torch_ms = measure(lambda: torch.nn.functional.layer_norm(x, (hidden_size,), weight, bias), device)
    custom_ms = measure(lambda: naive_layernorm(x, weight, bias), device)

    return BenchResult(
        hidden_size=hidden_size,
        max_error=max_error,
        torch_ms=torch_ms,
        custom_ms=custom_ms,
        speedup=torch_ms / custom_ms if custom_ms > 0 else 0.0,
    )


def naive_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    normalized = (x - mean) * torch.rsqrt(variance + eps)
    return normalized * weight + bias


def measure(fn, device: str) -> float:
    for _ in range(WARMUP_ITERS):
        fn()
    sync(device)

    start = time.perf_counter()
    for _ in range(BENCH_ITERS):
        fn()
    sync(device)

    return (time.perf_counter() - start) * 1000 / BENCH_ITERS


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def write_report(results: list[BenchResult], device: str):
    lines = [
        "# LayerNorm Kernel Benchmark",
        "",
        "## Setup",
        "",
        f"- Device: `{device}`",
        f"- Rows: `{ROWS}`",
        f"- Warmup iterations: `{WARMUP_ITERS}`",
        f"- Benchmark iterations: `{BENCH_ITERS}`",
        "- Baseline: `torch.nn.functional.layer_norm`",
        "- Custom path: small explicit PyTorch implementation of LayerNorm math",
        "",
        "## Results",
        "",
        "| Hidden size | Max error | Torch time | Custom time | Speedup |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in results:
        lines.append(
            f"| {result.hidden_size} | {result.max_error:.6f} | {result.torch_ms:.4f} ms | "
            f"{result.custom_ms:.4f} ms | {result.speedup:.2f}x |"
        )

    lines.extend(
        [
            "",
            "## What This Shows",
            "",
            "- Correctness comes first: the custom math is compared with PyTorch by max error.",
            "- Benchmarking needs warmup and synchronization, especially on CUDA.",
            "- This file is a minimal kernel-development loop. Replacing `naive_layernorm` with Triton or CUDA is the next step.",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
