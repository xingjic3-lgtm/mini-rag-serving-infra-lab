from __future__ import annotations

from pathlib import Path
import importlib.util
import platform
import shutil
import subprocess
import sys

import torch


REPORT_PATH = Path(__file__).with_name("vllm_env_report.md")


def main():
    checks = collect_checks()
    print("[vLLM Environment Check]")
    for name, value in checks.items():
        print(safe_console_text(f"{name}: {value}"))

    write_report(checks)
    print(f"\nReport written to: {REPORT_PATH}")


def collect_checks() -> dict[str, str]:
    return {
        "python": sys.executable,
        "platform": platform.platform(),
        "vllm_module": "found" if importlib.util.find_spec("vllm") else "missing",
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "wsl": command_status(["wsl.exe", "-l", "-v"]),
        "docker": "found" if shutil.which("docker") else "missing",
    }


def command_status(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, timeout=10)
    except FileNotFoundError:
        return "missing"
    except subprocess.TimeoutExpired:
        return "timeout"

    if result.returncode == 0:
        return "available"
    output = decode_process_output((result.stdout or b"") + (result.stderr or b"")).strip().replace("\n", " ")
    return f"unavailable: {output[:180]}"


def decode_process_output(output: bytes) -> str:
    if not output:
        return ""
    if b"\x00" in output[:80]:
        return output.decode("utf-16le", errors="replace")
    return output.decode("utf-8", errors="replace")


def write_report(checks: dict[str, str]):
    lines = [
        "# vLLM Environment Report",
        "",
        "| Check | Value |",
        "| --- | --- |",
    ]
    for name, value in checks.items():
        escaped = value.replace("|", "\\|")
        lines.append(f"| {name} | `{escaped}` |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Official vLLM GPU serving is Linux-first and is not supported in this native Windows Python environment.",
            "- Use WSL/Linux, Docker with a vLLM backend, or a remote Linux GPU host for the actual vLLM server.",
            "- Keep `vllm_client.py` on Windows if the server is reachable at `http://127.0.0.1:8000/v1`, or run both server and client inside Linux.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_console_text(text: str) -> str:
    return text.encode("gbk", errors="replace").decode("gbk")


if __name__ == "__main__":
    main()
