#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-1.5B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-local-token}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

vllm serve "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --dtype auto
