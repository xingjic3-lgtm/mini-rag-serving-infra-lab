# mini-rag-serving-infra-lab

一个围绕本地文档问答场景构建的 LLM serving 性能观测与机制学习项目。

项目不是单纯部署一个大模型，也不是声称已经完成生产级优化；它更关注一条真实 AI Infra 工作链路：

```text
业务请求 -> RAG prompt -> 本地 LLM 推理 -> benchmark -> request trace
       -> KV cache 分析 -> prefix cache 复用 -> vLLM serving 接入检查
       -> profiler 观测 -> distributed inference toy
```

目标是把 LLM serving 里的关键对象落到可运行代码和报告里：TTFT、TPOT、tokens/s、GPU memory、prefill、decode、KV cache、prefix cache、profile trace、tensor parallel communication。

## What This Project Shows

- 如何把文档问答业务请求转换成 LLM prompt。
- 如何建立 Transformers baseline，并记录 TTFT / TPOT / throughput / GPU memory。
- 如何追踪一次请求从 queue、prefill、decode 到 finished 的生命周期。
- 如何检查 KV cache 的真实 tensor shape 和显存增长。
- 如何用 prefix cache 复用相同文档前缀，观察 TTFT 变化。
- 如何准备接入 vLLM OpenAI-compatible serving API，并在当前环境不可用时给出明确 blocked 证据。
- 如何用 PyTorch profiler 标记 prefill / decode，并导出 trace。
- 如何用一个 LayerNorm toy 演示 correctness check + benchmark 方法。
- 如何用 `torch.distributed` toy 理解 tensor parallel 的权重切分和通信合并。

## Repository Layout

```text
mini-rag-serving-infra-lab/
├── 01_business_app/              # 最小 RAG 业务闭环
├── 02_baseline_benchmark/        # Transformers baseline 性能测试
├── 03_request_scheduler/         # 请求生命周期 trace
├── 04_kv_cache_analysis/         # KV cache shape / memory 分析
├── 05_prefix_cache/              # prefix cache reuse 实验
├── 06_vllm_serving/              # vLLM server/client 接入与环境检查
├── 07_kernel_profile/            # prefill/decode profile + LayerNorm benchmark
├── 08_distributed_inference/     # tensor parallel Linear toy
├── reports/                      # 跨阶段报告
├── models/                       # 本地模型目录，默认不提交
└── run_phase*.ps1                # Windows PowerShell 运行脚本
```

## Current Environment

当前实验使用：

- Model: `models/Qwen2.5-1.5B-Instruct`
- Backend baseline: Transformers local generation
- Python env used in scripts: `D:\anaconda3\envs\cuda-lab\python.exe`
- GPU observed by PyTorch: `NVIDIA GeForce RTX 5060 Ti`

`models/` 是本地模型权重目录，已在 `.gitignore` 中排除。

## Results Snapshot

### Baseline

From `02_baseline_benchmark/baseline_report.md`:

| Metric | Value |
| --- | ---: |
| Prompt length | 171 tokens |
| Output length | 21 tokens |
| TTFT | 0.0301 s |
| TPOT | 0.0261 s/token |
| Throughput | 38.04 tokens/s |
| Peak GPU Memory | 2.96 GB |
| Total latency | 0.5520 s |

### Request Lifecycle

From `reports/request_lifecycle.md`:

```text
enter queue -> preprocess -> prefill -> decode step by step -> finished
```

The trace records prompt length, generated token per decode step, elapsed time, and GPU memory.

### KV Cache

From `04_kv_cache_analysis/kv_cache_report.md`:

| Prompt length | Estimated KV cache |
| ---: | ---: |
| 512 tokens | 14.00 MB |
| 2048 tokens | 56.00 MB |
| 8192 tokens | 224.00 MB |

The measured current request KV cache is `4.6758 MB` for `seq_len=171`.

### Prefix Cache

From `05_prefix_cache/reuse_report.md`:

| Metric | Value |
| --- | ---: |
| Cached prefix tokens | 1381 |
| Request suffix tokens | 8 |
| TTFT without prefix reuse | 0.2308 s |
| TTFT with prefix reuse | 0.1503 s |
| TTFT speedup | 1.54x |

### vLLM Serving

From `06_vllm_serving/vllm_benchmark.md`:

```text
Status: blocked
Reason: no reachable vLLM server in the current native Windows environment.
```

The client is wired to the OpenAI-compatible endpoint:

```text
http://127.0.0.1:8000/v1/chat/completions
```

It does not write fake vLLM numbers. To produce a real comparison, run vLLM in WSL/Linux or on a Linux GPU host and rerun the client.

### Profile

From `07_kernel_profile/profile_report.md`:

- Torch profiler trace: `07_kernel_profile/nsight_trace/torch_profiler_trace.json`
- Current note: CUDA is available, but CUPTI did not provide CUDA activity timing in this run.
- The code marks `prefill` and `decode` with `record_function`.

From `07_kernel_profile/layernorm_benchmark.md`:

| Hidden size | Max error | Torch time | Custom time | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 0.000002 | 0.0104 ms | 0.0746 ms | 0.14x |
| 1024 | 0.000001 | 0.0106 ms | 0.0830 ms | 0.13x |
| 2048 | 0.000002 | 0.0122 ms | 0.0969 ms | 0.13x |
| 4096 | 0.000002 | 0.0173 ms | 0.1243 ms | 0.14x |

The custom LayerNorm is intentionally simple and slower than PyTorch. Its purpose is correctness and benchmark methodology, not kernel optimization.

## How To Run

Install Python dependencies:

```bash
pip install -r requirements.txt
```

On this Windows setup, the provided scripts use:

```text
D:\anaconda3\envs\cuda-lab\python.exe
```

Run each phase:

```powershell
.\run_phase01.ps1
.\run_phase02_benchmark.ps1
.\run_phase03_trace.ps1
.\run_phase04_kv_cache.ps1
.\run_phase05_prefix_cache.ps1
.\run_phase06_vllm_client.ps1
.\run_phase07_profile.ps1
.\run_phase08_distributed.ps1
```

Or run Python files directly:

```bash
python 01_business_app/app.py
python 02_baseline_benchmark/benchmark.py
python 03_request_scheduler/request_trace.py
python 04_kv_cache_analysis/kv_cache_trace.py
python 05_prefix_cache/prefix_cache_demo.py
python 06_vllm_serving/vllm_env_check.py
python 06_vllm_serving/vllm_client.py
python 07_kernel_profile/profile_torch_decode.py
python 07_kernel_profile/layernorm_kernel.py
python 08_distributed_inference/tp_linear_demo.py
```

## vLLM Notes

The project includes:

- `06_vllm_serving/vllm_server.sh`
- `06_vllm_serving/vllm_client.py`
- `06_vllm_serving/vllm_env_check.py`

Current native Windows execution is blocked because official vLLM GPU serving is Linux-first. Use WSL/Linux, Docker, or a remote Linux GPU host:

```bash
cd /mnt/d/Project/cudaproject/project
bash 06_vllm_serving/vllm_server.sh
python 06_vllm_serving/vllm_client.py
```

When the server is reachable, `vllm_client.py` will replace the blocked report with a real Transformers vs vLLM comparison.

## Learning Focus

This repo is best described as:

```text
LLM serving performance observation and AI Infra mechanism lab
```

It is useful for learning:

- how business requests become model inference requests;
- how TTFT, TPOT, throughput, and memory are measured;
- how prefill and decode differ;
- what KV cache looks like as real tensors;
- why prefix reuse can reduce TTFT;
- how serving engines such as vLLM expose model inference as an API;
- how profiler traces are collected and interpreted;
- how tensor parallel toy examples split computation and communicate results.

## Limitations

- The vLLM server has not been executed in this native Windows environment.
- The profile report currently lacks CUDA activity timing because CUPTI did not provide CUDA events in this run.
- The distributed inference phase is a toy Linear example, not a full multi-GPU LLM deployment.
- The project currently emphasizes measurement and mechanism understanding more than production optimization.

## Resume-Friendly Summary

Built a mini RAG serving performance lab around Qwen2.5-1.5B, covering baseline benchmarking, request lifecycle tracing, KV cache memory analysis, prefix cache reuse, vLLM serving integration checks, PyTorch profiler tracing, LayerNorm benchmark methodology, and a tensor-parallel Linear toy demo using `torch.distributed`.
