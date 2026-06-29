# mini-rag-serving-infra-lab 项目计划书

## 1. 项目名称

**mini-rag-serving-infra-lab：面向文档问答业务的 LLM Serving Infra 实验项目**

---

## 2. 项目定位

这不是一个单纯“部署大模型”的练习。

本项目模拟真实 AI Infra 工作场景：

> 公司要上线一个企业文档问答助手，用户上传文档后可以提问。初始版本能跑，但速度慢、显存占用高、并发能力差。AI Infra 工程师需要通过 benchmark、profile、KV Cache 分析、scheduler 优化、prefix cache、vLLM 接入、多卡实验等方式，让系统变快、变稳、变便宜。

---

## 3. 业务闭环

完整链路如下：

```text
用户上传文档
↓
系统切分文档
↓
建立向量索引
↓
用户提出问题
↓
RAG 检索相关片段
↓
拼接 prompt
↓
LLM 生成答案
↓
记录 TTFT / TPOT / tokens/s / 显存 / 并发失败率

```

项目最终要证明：

> 同一个文档问答业务，在 AI Infra 优化前后，推理速度、显存占用、并发能力有明确变化。

---

## 4. 最终成果

最终 GitHub 仓库必须包含：

```text
mini-rag-serving-infra-lab/
├── 01_business_app/
├── 02_baseline_benchmark/
├── 03_request_scheduler/
├── 04_kv_cache_analysis/
├── 05_prefix_cache/
├── 06_vllm_serving/
├── 07_kernel_profile/
├── 08_distributed_inference/
├── reports/
└── README.md
```

最终 README 必须能说明：

```text
1. 这个系统解决什么业务问题
2. 请求从进入系统到生成答案经历了什么
3. baseline 性能是多少
4. 慢在哪里
5. 做了哪些 AI Infra 优化
6. 优化后指标变成多少
7. 每个实验如何复现
```

---

# 5. 项目阶段规划

## Phase 01：业务系统最小闭环

### 真实工单

> 做一个本地文档问答助手，用户可以输入一段文档，然后提问，系统调用本地 LLM 生成回答。

### 核心产出

```text
01_business_app/app.py
01_business_app/rag_pipeline.py
01_business_app/README.md
```

### 最小功能

```text
1. 输入一段文档
2. 输入一个问题
3. 检索相关文本片段
4. 拼接 prompt
5. 调用本地模型生成答案
6. 打印最终回答
```

### 完成标准

运行：

```bash
python 01_business_app/app.py
```

可以看到：

```text
[Document Loaded]
[Question]
[Retrieved Context]
[Prompt]
[LLM Answer]
```

### 这一阶段学到的工作能力

```text
业务请求如何变成 LLM 推理请求
RAG 和 LLM serving 的关系
模型在业务系统里的真实作用
```

---

## Phase 02：建立 baseline benchmark

### 真实工单

> 当前文档问答系统能跑，但不知道慢在哪里。先建立性能基线。

### 核心产出

```text
02_baseline_benchmark/benchmark.py
02_baseline_benchmark/baseline_report.md
```

### 必须统计的指标

```text
TTFT：Time To First Token，首 token 延迟
TPOT：Time Per Output Token，每个输出 token 平均耗时
tokens/s：生成吞吐
peak GPU memory：峰值显存
total latency：总延迟
```

### 完成标准

运行：

```bash
python 02_baseline_benchmark/benchmark.py
```

输出类似：

```text
Prompt length: 1024 tokens
Output length: 128 tokens
TTFT: 2.41s
TPOT: 0.036s/token
Throughput: 27.7 tokens/s
Peak GPU Memory: 9.8 GB
```

### 这一阶段学到的工作能力

```text
不是凭感觉说慢，而是用指标描述慢
建立 baseline
为后续优化提供对比对象
```

---

## Phase 03：请求生命周期追踪

### 真实工单

> 业务请求进入系统后到底经历了什么不清楚，需要把请求生命周期打出来。

### 核心产出

```text
reports/request_lifecycle.md
03_request_scheduler/request_trace.py
```

### 必须追踪的状态

```text
request_id
prompt token 数
进入队列时间
开始 prefill 时间
开始 decode 时间
每一步 decode 生成的 token
结束时间
显存变化
```

### 完成标准

运行：

```bash
python 03_request_scheduler/request_trace.py
```

可以看到：

```text
[request_001] enter queue
[request_001] prefill start, prompt_len=856
[request_001] decode step=1, new_token=...
[request_001] decode step=2, new_token=...
[request_001] finished, total_latency=...
```

### 这一阶段学到的工作能力

```text
理解 request → scheduler → prefill → decode → output
理解服务系统不是单次 model.forward
```

---

## Phase 04：KV Cache 形状与显存分析

### 真实工单

> 长文档问答时显存涨得很快，需要分析 KV Cache 到底占了多少。

### 核心产出

```text
04_kv_cache_analysis/kv_cache_trace.py
04_kv_cache_analysis/kv_cache_report.md
```

### 必须打印的信息

```text
layer 数
num_kv_heads
head_dim
seq_len
每层 K shape
每层 V shape
单请求 KV Cache 显存估算
不同 prompt 长度下的显存增长表
```

### 完成标准

运行：

```bash
python 04_kv_cache_analysis/kv_cache_trace.py
```

输出类似：

```text
Layer 0:
K shape = [num_kv_heads, seq_len, head_dim]
V shape = [num_kv_heads, seq_len, head_dim]

Prompt len = 512, KV Cache = xxx MB
Prompt len = 2048, KV Cache = xxx MB
Prompt len = 8192, KV Cache = xxx MB
```

### 这一阶段学到的工作能力

```text
KV Cache 不是抽象概念，而是显存里的真实张量
长上下文为什么吃显存
decode 为什么依赖历史 KV
```

---

## Phase 05：实现朴素 KV Cache Manager

### 真实工单

> 多个用户同时请求时，每个请求都需要维护自己的 KV Cache。先实现一个 toy 版 KV Cache Manager。

### 核心产出

```text
04_kv_cache_analysis/naive_kv_cache_manager.py
```

### 必须支持

```text
create_request(request_id)
append_token_kv(request_id, layer_id, k, v)
get_cache(request_id)
free_request(request_id)
print_cache_table()
```

### 完成标准

运行：

```bash
python 04_kv_cache_analysis/naive_kv_cache_manager.py
```

可以看到：

```text
request_001: seq_len=3, layers=...
request_002: seq_len=5, layers=...
free request_001
request_002 still alive
```

### 这一阶段学到的工作能力

```text
KV Cache 是按 request 管理的资源
请求结束后必须释放 cache
多请求场景下显存管理开始变复杂
```

---

## Phase 06：实现 Paged KV Cache toy

### 真实工单

> 朴素 KV Cache 连续分配容易浪费显存。实现 block/page 式 KV Cache 管理，模拟 vLLM 的核心思想。

### 核心产出

```text
04_kv_cache_analysis/paged_kv_cache_demo.py
04_kv_cache_analysis/block_table_report.md
```

### 必须支持

```text
固定 block_size
free block pool
request_id → block list 映射
append token 时自动申请 block
request 结束时释放 block
打印 block table
```

### 完成标准

运行：

```bash
python 04_kv_cache_analysis/paged_kv_cache_demo.py
```

输出类似：

```text
Free blocks: [5, 6, 7, 8]
request_001 -> blocks [0, 1]
request_002 -> blocks [2, 3, 4]

free request_001

Free blocks: [0, 1, 5, 6, 7, 8]
request_002 -> blocks [2, 3, 4]
```

### 这一阶段学到的工作能力

```text
理解 PagedAttention 为什么像操作系统分页
理解 block table
理解显存碎片和 block 复用
```

---

## Phase 07：实现简单 Request Scheduler

### 真实工单

> 多个用户请求同时到来，不能一个一个跑。需要把请求组成 batch，提高 GPU 利用率。

### 核心产出

```text
03_request_scheduler/scheduler_demo.py
03_request_scheduler/scheduler_report.md
```

### 必须支持

```text
waiting_queue
running_queue
finished_queue
每轮选择一批 request
区分 prefill 和 decode
打印每个 timestep 的 batch
```

### 完成标准

运行：

```bash
python 03_request_scheduler/scheduler_demo.py
```

输出类似：

```text
t=0 batch: request_001 prefill, request_002 prefill
t=1 batch: request_001 decode, request_002 decode, request_003 prefill
t=2 batch: request_001 decode, request_002 finished, request_003 decode
```

### 这一阶段学到的工作能力

```text
continuous batching 的基本矛盾
不同长度请求为什么会影响吞吐
scheduler 如何决定谁进入 batch
```

---

## Phase 08：Prefix Cache 复用实验

### 真实工单

> 文档问答系统里很多请求有相同 system prompt 或相同文档前缀。重复 prefill 浪费算力，需要做 prefix cache reuse 实验。

### 核心产出

```text
05_prefix_cache/prefix_cache_demo.py
05_prefix_cache/reuse_report.md
```

### 必须支持

```text
识别相同 prefix
缓存 prefix 对应的 KV
新请求命中 prefix 时跳过重复 prefill
打印 cache hit / miss
对比命中前后的 TTFT
```

### 完成标准

运行：

```bash
python 05_prefix_cache/prefix_cache_demo.py
```

输出类似：

```text
request_001: prefix miss, run prefill
request_002: prefix hit, reuse 512 tokens KV
TTFT before reuse: 2.3s
TTFT after reuse: 0.9s
```

### 这一阶段学到的工作能力

```text
理解为什么多轮对话和系统提示词适合 prefix cache
理解 KV Cache 不只是保存历史 token，也可以跨请求复用
```

---

## Phase 09：接入 vLLM 做真实 serving 对比

### 真实工单

> toy 系统已经理解了，现在接入真实推理框架，看 vLLM 相比 transformers baseline 的性能变化。

### 核心产出

```text
06_vllm_serving/vllm_server.sh
06_vllm_serving/vllm_client.py
06_vllm_serving/vllm_benchmark.md
```

### 必须对比

```text
transformers baseline
vLLM serving
同一批 prompt
同一输出长度
TTFT
TPOT
tokens/s
peak GPU memory
并发请求数
```

### 完成标准

运行：

```bash
bash 06_vllm_serving/vllm_server.sh
python 06_vllm_serving/vllm_client.py
```

得到对比表：

```text
Engine          TTFT      TPOT      tokens/s      Peak Memory
transformers    ...
vLLM            ...
```

### 这一阶段学到的工作能力

```text
会部署真实 LLM serving 框架
会做 engine 对比
会用数据解释为什么 vLLM 更适合 serving
```

---

## Phase 10：vLLM 源码观测：Block 分配日志

### 真实工单

> 线上出现显存问题，需要进入 vLLM 源码观察 block 分配、释放和复用行为。

### 核心产出

```text
06_vllm_serving/vllm_block_trace.patch
06_vllm_serving/block_trace_log.md
```

### 必须观察

```text
request 到来时申请了哪些 block
decode 增长时是否申请新 block
request 结束时 block 是否释放
prefix cache 是否命中
```

### 完成标准

运行压测后可以看到类似日志：

```text
[ALLOC] request_id=xxx block_id=12
[ALLOC] request_id=xxx block_id=13
[FREE] request_id=xxx block_id=12
[REUSE] prefix block hit
```

### 这一阶段学到的工作能力

```text
从 toy 版本迁移到真实框架源码
能定位真实系统里的 KV Cache 管理位置
能给真实框架打日志做观测
```

---

## Phase 11：Kernel Profile

### 真实工单

> vLLM 接入后仍然有瓶颈，需要用 profiling 工具确认时间花在哪些 kernel 上。

### 核心产出

```text
07_kernel_profile/profile_report.md
07_kernel_profile/nsight_trace/
```

### 必须分析

```text
prefill 阶段主要 kernel
decode 阶段主要 kernel
attention kernel 时间占比
matmul kernel 时间占比
GPU utilization
memory bandwidth
是否存在 CPU 调度空洞
```

### 完成标准

报告里必须有：

```text
1. profile 命令
2. 截图或日志
3. Top 5 kernel 耗时
4. 当前瓶颈判断
5. 下一步优化建议
```

### 这一阶段学到的工作能力

```text
用 profile 证据定位瓶颈
区分 CPU overhead / GPU kernel / memory bandwidth / communication
```

---

## Phase 12：写一个 Triton/CUDA 小算子

### 真实工单

> 某些场景下框架默认算子不够快，需要具备写 kernel 和 benchmark 的能力。

### 核心产出

```text
07_kernel_profile/layernorm_kernel.py
07_kernel_profile/layernorm_benchmark.md
```

### 必须包含

```text
PyTorch baseline
Triton 或 CUDA 实现
correctness check
benchmark
不同 hidden size 的性能表
```

### 完成标准

运行：

```bash
python 07_kernel_profile/layernorm_kernel.py
```

输出：

```text
max error: < 1e-3
torch time: ...
custom kernel time: ...
speedup: ...
```

### 这一阶段学到的工作能力

```text
算子正确性验证
benchmark 方法
GPU kernel 最小开发闭环
```

---

## Phase 13：多卡推理 toy

### 真实工单

> 单卡显存不够，需要理解 tensor parallel 的基本机制。

### 核心产出

```text
08_distributed_inference/tp_linear_demo.py
08_distributed_inference/distributed_report.md
```

### 必须实现

```text
用 torchrun 启动 2 个进程
每个进程持有一部分 Linear 权重
各自计算 partial output
通过 all-gather 或 all-reduce 合并结果
和单卡完整 Linear 对比误差
```

### 完成标准

运行：

```bash
torchrun --nproc_per_node=2 08_distributed_inference/tp_linear_demo.py
```

输出：

```text
rank 0 weight shard shape = ...
rank 1 weight shard shape = ...
merged output shape = ...
max error = ...
```

### 这一阶段学到的工作能力

```text
理解 tensor parallel 不是概念，而是权重切分 + 通信合并
理解多卡推理为什么会遇到通信瓶颈
```

---

# 6. 最终报告要求

最终必须写：

```text
reports/final_report.md
```

内容必须包含：

```text
1. 业务背景
2. 系统架构
3. 请求生命周期
4. baseline 性能
5. 发现的瓶颈
6. 做过的优化
7. 优化前后指标对比
8. 失败尝试
9. 项目总结
10. 下一步可扩展方向
```

最终对比表格式：

```text
版本              TTFT      TPOT      tokens/s      Peak Memory      Max Concurrency
baseline          ...
+ scheduler        ...
+ prefix cache     ...
+ vLLM             ...
```

---

# 7. 项目执行规则

带做 AI 必须遵守：

```text
1. 不要一次性完成整个项目
2. 每次只带我做一个小步骤
3. 每一步必须说明当前这一步属于哪个 Phase
4. 每一步必须告诉我：
   - 当前要做什么
   - 在 VS Code 创建哪个文件或文件夹
   - 文件路径是什么
   - 文件里写什么代码
   - 终端输入什么命令
   - 成功后应该看到什么
5. 写代码时必须逐行解释关键变量：
   - 这个变量代表真实系统里的什么对象
   - 它的 shape / 类型 / 值是什么
   - 它为什么要这样计算
6. 不要直接把所有代码写完
7. 不要自动扩展功能
8. 不要引入复杂架构
9. 每一步结束后停下来，等我回复“完成”或贴报错
10. 如果报错，先判断是环境问题、代码问题、依赖问题还是理解问题，再处理
```

---

# 8. 每一步输出模板

带做 AI 每一步必须按这个格式输出：

````text
当前 Phase：
当前小任务：
这一小步的目标：

你现在要创建/修改的文件：
文件路径：

写入代码：
```python
...
````

关键代码解释：

1. xxx 变量：

   * 真实对象：
   * 类型/shape：
   * 为什么这样写：

2. xxx 变量：

   * 真实对象：
   * 类型/shape：
   * 为什么这样写：

终端命令：

```bash
...
```

成功后你应该看到：

```text
...
```

本步完成标准：
看到 xxx 就算完成。

停在这里。
你完成后回复：完成
如果报错，把完整报错贴出来。

````

---

# 9. 禁止事项

带做 AI 不允许：

```text
1. 一次性生成整个项目
2. 一次性给多个 Phase
3. 上来讲大量概念
4. 写没有业务闭环的 toy demo
5. 跳过 benchmark
6. 只说“优化了”，但不给指标
7. 只讲 KV Cache 概念，不落到 request / block / tensor / 显存
8. 把环境问题当成 bug 长篇修
9. 私自改项目目标
10. 私自增加复杂功能
````

---

# 10. 第一阶段启动任务

项目从 Phase 01 开始。

第一步只做：

```text
创建项目目录结构
创建 01_business_app/app.py
让 app.py 先跑通一个最小业务闭环：
输入固定文档 + 固定问题 → 拼接 prompt → 打印 prompt
暂时不接模型
```

第一步完成标准：

运行：

```bash
python 01_business_app/app.py
```

看到：

```text
[Document Loaded]
[Question]
[Retrieved Context]
[Prompt]
```

注意：

```text
第一步不要安装模型
第一步不要接 vLLM
第一步不要写 KV Cache
第一步只让业务链路骨架跑起来
```

---

# 11. 给 AI 的启动提示词

请你从现在开始进入项目带做模式。

你不是自动 coding agent，而是我的 AI Infra 工程老师。

请严格按照《mini-rag-serving-infra-lab 项目计划书》带我完成项目。

当前只做 Phase 01 的第一个小步骤：

> 创建项目目录结构，创建 `01_business_app/app.py`，让它跑通最小业务闭环：固定文档 + 固定问题 → 检索固定片段 → 拼接 prompt → 打印 prompt。暂时不接模型。

规则：

1. 不要一次性完成整个项目。
2. 不要主动扩展功能。
3. 每次只给我一个小步骤。
4. 每一步必须告诉我：

   * 当前要做什么
   * 在 VS Code 创建哪个文件或文件夹
   * 文件路径是什么
   * 文件里写什么代码
   * 终端输入什么命令
   * 成功后应该看到什么
5. 写代码时必须解释关键变量的真实对象含义、类型/shape、为什么这样写。
6. 每一步结束后停下来，等我回复“完成”或贴报错。
7. 不要跳到模型部署、KV Cache、vLLM、CUDA。
8. 当前第一步只完成最小业务闭环骨架。

现在开始带我做第一步。
