# vLLM Environment Report

| Check | Value |
| --- | --- |
| python | `D:\anaconda3\envs\cuda-lab\python.exe` |
| platform | `Windows-10-10.0.26200-SP0` |
| vllm_module | `missing` |
| torch | `2.12.0.dev20260408+cu128` |
| cuda_available | `True` |
| cuda_device | `NVIDIA GeForce RTX 5060 Ti` |
| wsl | `unavailable: 未安装适用于 Linux 的 Windows 子系统。可通过运行 “wsl.exe --install” 进行安装。 有关详细信息，请访问 https://aka.ms/wslinstall` |
| docker | `missing` |

## Interpretation

- Official vLLM GPU serving is Linux-first and is not supported in this native Windows Python environment.
- Use WSL/Linux, Docker with a vLLM backend, or a remote Linux GPU host for the actual vLLM server.
- Keep `vllm_client.py` on Windows if the server is reachable at `http://127.0.0.1:8000/v1`, or run both server and client inside Linux.
