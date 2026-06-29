$ErrorActionPreference = "Stop"

$python = "D:\anaconda3\envs\cuda-lab\python.exe"
& $python "06_vllm_serving\vllm_env_check.py"
& $python "06_vllm_serving\vllm_client.py"
