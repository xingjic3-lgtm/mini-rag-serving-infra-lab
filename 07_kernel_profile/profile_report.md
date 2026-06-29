# Kernel Profile Report

## Profile Command

```bash
python 07_kernel_profile/profile_torch_decode.py
```

Optional Nsight command:

```bash
nsys profile -o 07_kernel_profile/nsight_trace/vllm_or_transformers_trace python 07_kernel_profile/profile_torch_decode.py
```

## Trace Artifact

- Torch profiler Chrome trace: `D:/Project/cudaproject/project/07_kernel_profile/nsight_trace/torch_profiler_trace.json`
- Device note: CUDA is available, but CUPTI did not provide CUDA activity timing in this run.

## Top Operators

```text
--------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                        Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
--------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                     prefill         2.80%      19.448ms        46.91%     325.315ms     325.315ms             1  
                             aten::embedding         0.02%     118.600us        12.16%      84.287ms       9.365ms             9  
                               aten::reshape         0.42%       2.915ms         0.55%       3.809ms       1.653us          2304  
                                  aten::view         0.28%       1.921ms         0.28%       1.921ms       0.400us          4806  
                          aten::index_select        11.56%      80.163ms        12.13%      84.114ms       9.346ms             9  
                                 aten::empty         0.19%       1.339ms         0.19%       1.339ms       1.381us           970  
                               aten::resize_         0.05%     319.400us         0.05%     319.400us       6.943us            46  
                                aten::expand         0.62%       4.290ms         0.76%       5.286ms       2.302us          2296  
                            aten::as_strided         0.46%       3.212ms         0.46%       3.212ms       0.434us          7402  
                                aten::gather         0.53%       3.684ms         0.53%       3.686ms       3.686ms             1  
                                aten::arange         0.35%       2.407ms         0.70%       4.855ms     269.706us            18  
                                   aten::add         2.90%      20.090ms         2.90%      20.090ms      13.130us          1530  
--------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 693.422ms

```

## Bottleneck Judgment

- Prefill processes the full prompt, so attention and matrix multiplication work scale with prompt length.
- Decode processes one new token at a time but repeatedly reads historical KV cache.
- If CUDA time is dominated by matrix multiplication operators, the current bottleneck is compute-heavy model layers.
- If CUDA time is dominated by attention or cache-related kernels, the current bottleneck is KV-cache read bandwidth and attention scheduling.
- If CPU time is large while CUDA time is small, the current bottleneck is request scheduling or Python/runtime overhead.

## Next Optimization Direction

- Compare this trace with vLLM serving under concurrency.
- Check whether decode has many small kernels, which usually points to batching or launch-overhead problems.
- Use Nsight Systems when you need the CPU scheduling gap between requests and GPU kernels.
