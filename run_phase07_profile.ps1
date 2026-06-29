$ErrorActionPreference = "Stop"

$python = "D:\anaconda3\envs\cuda-lab\python.exe"
& $python "07_kernel_profile\profile_torch_decode.py"
& $python "07_kernel_profile\layernorm_kernel.py"
