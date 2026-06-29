$ErrorActionPreference = "Stop"

$env:MODEL_PATH = if ($env:MODEL_PATH) { $env:MODEL_PATH } else { "models/Qwen2.5-1.5B-Instruct" }
$env:HOST = if ($env:HOST) { $env:HOST } else { "127.0.0.1" }
$env:PORT = if ($env:PORT) { $env:PORT } else { "8000" }
$env:API_KEY = if ($env:API_KEY) { $env:API_KEY } else { "local-token" }
$env:MAX_MODEL_LEN = if ($env:MAX_MODEL_LEN) { $env:MAX_MODEL_LEN } else { "2048" }
$env:GPU_MEMORY_UTILIZATION = if ($env:GPU_MEMORY_UTILIZATION) { $env:GPU_MEMORY_UTILIZATION } else { "0.85" }

vllm serve $env:MODEL_PATH `
  --host $env:HOST `
  --port $env:PORT `
  --api-key $env:API_KEY `
  --max-model-len $env:MAX_MODEL_LEN `
  --gpu-memory-utilization $env:GPU_MEMORY_UTILIZATION `
  --dtype auto
