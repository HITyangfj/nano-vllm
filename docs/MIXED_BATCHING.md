# Decode 优先混合批处理

## 范围和配置

`enable_mixed_batching=True` 默认开启，`False` 为 Prefill 优先的分阶段对比模式。
两个模式共用执行引擎、缓存和采样逻辑。关闭开关并不等于原始 revision：关闭模式
也修复了 Decode 不扣 token budget、轮转公平性和无采样 chunk 等问题。
真正的原始版本对照需使用独立的 `bb823b3e06983d71485a8e1f23715ebd87d98ef8` checkout。

`max_num_seqs` 约束**本轮 batch 的不同请求数**，不限制 waiting/running 队列总数。
`max_num_batched_tokens` 同时计入 Decode 和 Prefill 的实际 query token。
有限请求集会通过轮转和资源回收推进；无限 Decode 流持续占满预算时，新 Prefill
可能等待，本策略不承诺无条件公平。

Prompt KV 仍按整个已知请求一次性分配，没有实现增量 Prompt KV 分配。
新请求分配失败会跳过，以便运行有缓存的请求。Decode 缺少下一块时，优先抢占
尚未选中的 running，其次释放尚未选中的部分 Prefill；本轮已选请求不会被回收。
若单个请求在其最大输出长度内所需 KV 超过整个池，入队时直接报可读错误。
校验 `prompt_len + max_tokens <= max_model_len`；空 Prompt 和非正 max_tokens 报错。

新增内存设置：

- `num_kvcache_blocks=-1` 自动估算；正数固定物理池容量，便于 A/B 对齐。
- `cuda_graph_memory_reserve=268435456` 在自动容量估计中为 Graph 预留 256 MiB。
  这是可调整的预留值，不是已测得的 Graph 峰值；较大模型/批次应在目标卡上核对。
- Warmup 在分配真实 KV 前执行，覆盖满 token budget、最大采样行数和长 query
  两类形状；仍使用无 cache/block table 的 varlen 路径。
- TP 池容量取所有 rank 的最小值；显式容量超过估计可用空间时拒绝启动。

## 执行和状态不变量

`WAITING` 包含新请求、部分 Prefill 和抢占重算。`RUNNING` 仅包含普通 Decode。
部分 Prefill 不转 RUNNING；只有末块完成并采样后转 RUNNING，EOS/max_tokens 后转 FINISHED。
`num_cached_tokens` 是已经计算或复用的前缀长度；`num_scheduled_tokens` 是本轮保留的计算量。
`num_computed_tokens` 保留抢占前的计算高水位，仅用于识别重复计算量。

`ScheduledRequest` 是不可变快照，含 seq_id、start/end、当时总长度、本轮实际 token、
block table、temperature 和请求阶段。SchedulerOutput 经 pickle 传给 TP worker，
worker 不需要切片 Sequence 的历史。Sequence 单独序列化也保留全部实例字段。
TP 沿用原来的共享内存通信；超过 1 MiB 的消息会报明确错误，并未重构传输机制。

输入在一次 forward 内拼接，positions 保持绝对位置 `[start,end)`。
slot_mapping 对位置 p 使用 `block_table[p // block_size] * block_size + p % block_size`。
query 边界累加实际 query 长度；每行有效 KV 长度为 end，而非分配容量。
先用 Triton 写 K/V，再用 FlashAttention varlen + paged block table 读取，
不向后端额外传 K/V 来自动追加。混合请求复用同一次投影、MLP 和模型 forward。
因果掩码按右下对齐，对局部第 j 个 query 只允许读取 `key_position <= start+j`。

只有 `end == 本轮开始时总长度` 的请求才进入 logits_indices/temperature/采样列表。
LM Head 显式选取这些请求最后一行，结果以 seq_id 映射提交；部分 chunk 不追加输出，
无采样批次在所有 rank 同时跳过 LM Head/gather/Sampler。rank 0 采样后 `.tolist()`
将结果取回 CPU；无采样 chunk 在 rank 0 同步完成后返回。
hash_blocks 在推进缓存进度之前发布新完成的完整块，不发布未来 slot。
重算追赶已有历史时不重复追加已有 completion。

纯 Decode 根据显式阶段选择 `flash_attn_with_kvcache` 与 CUDA Graph，不能用
“所有 q_len 都为 1”推断。Graph bucket 只捕获 buffer 容量内的大小，并覆盖配置上界；
超出 bucket 或 block table 宽度时 eager 回退。padding slot=-1、context_len=0，
每次复制前清理残留 block table。LM Head 在 graph 外按实际行采样。

## CPU 验收示例

执行 `python tools/inspect_mixed_schedule.py --output docs/validation_schedule.json`，
得到 [真实调度器生成的记录](validation_schedule.json)。采样值用固定 7 代替模型输出。

| C 的分块轮次 | A Decode | B Decode | C Prefill | C 的 start/end | C 是否采样 |
|---|---:|---:|---:|---|---|
| 1 | 1 | 1 | 510 | 0 / 510 | 否 |
| 2 | 1 | 1 | 510 | 510 / 1020 | 否 |
| 3 | 1 | 1 | 180 | 1020 / 1200 | 是 |

第一轮 query_start_loc 为 `[0,1,2,512]`，logits_indices 为 `[0,1]`；
第三轮为 `[0,1,2,182]` 和 `[0,1,181]`。第一轮有 512 个 slot，对应 hidden states
`[512,hidden_size]`，Q 为 `[512,num_query_heads,head_dim]`。

## 验证

开发环境：Windows、Python 3.10.9、PyTorch 2.0.0、GTX 1650 Ti 4 GiB，
有本地 Qwen3 权重；缺少 transformers、Triton、FlashAttention，GPU 不属于本路径
所需的 FlashAttention-2 Ampere/Ada/Hopper 范围。没有尝试把项目改为 Turing 后端。
本机没有执行 GPU 数值/Graph/TP/性能验证，不能据此宣称加速或无回归。

本机使用隔离 `.venv`（继承现有系统包，仅另外安装 xxhash）运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m compileall -q nanovllm tests bench_mixed.py bench_matrix.py tools
.\.venv\Scripts\python.exe tools/inspect_mixed_schedule.py --output docs/validation_schedule.json
git --git-dir=.git --work-tree=. diff --check
```

结果：60 passed，2 skipped（GPU 模块级跳过）；编译及 diff 空白检查通过。
CPU 覆盖调度、预算/轮转、255/256/257/511/512/513 块边界、Prefix Cache 引用计数、
抢占/重算、序列化、输入映射、真实 LM Head 行选择、引擎 step/generate、指标聚合，
以及用 CPU buffer 检查真实 runner 的 Graph padding/回退和无采样控制流。
CPU buffer 检查不代表 CUDA Graph 实机通过。

在兼容 Linux/CUDA 环境安装项目依赖和 pytest 后：

```bash
python -m pytest tests/test_attention_gpu.py -v
NANO_VLLM_TEST_MODEL=/path/to/Qwen3-0.6B python -m pytest tests/test_model_gpu.py -v -k 'not tp_two'
NANO_VLLM_TEST_MODEL=/path/to/Qwen3-0.6B NANO_VLLM_TEST_TP=2 \
  python -m pytest tests/test_model_gpu.py -v -k tp_two
```

Attention 测试用随机的物理页、GQA 和绝对位置 fp32 参考计算，容差 atol/rtol=3e-3。
模型测试以固定 token 历史比较 singleton、混批、前缀命中及重算，logits 容差
atol=5e-2、rtol=1e-2，需在目标 dtype/设备上确认。Graph 比较 1、3、9、16、17
及 18（17 容量以外的 eager 回退）。TP=2 单独运行，避免同时创建两个 NCCL group。
模型比较不依赖随机种子得到相同生成序列，也没有使用不被 SamplingParams 支持的 temperature=0。

## 固定轨迹性能评测

原 `bench.py` 保留可运行。`bench_mixed.py` 提供三类共享轨迹：

```bash
python bench_mixed.py --make-trace offline --count 256 --trace results/offline.json
python bench_mixed.py --make-trace injection --count 3 --injection-at 0.2 --trace results/injection.json
python bench_mixed.py --make-trace mixed --count 32 --trace results/arrivals.json
```

offline 沿用原短输入基准的长度分布；injection 是两个短请求后在固定墙钟时刻注入
长 Prompt；mixed 按固定时间分组到达长短请求。注入时间应先校准，使原请求已经开始
Decode 且尚未结束，再固定同一轨迹用于全部模式。结果中的 `injection_case_valid`
为 false 表示该次没有覆盖“两请求正在 Decode”的验收场景，需要调整轨迹重跑全部模式。
不能把不同算法各自调整后的不同到达时间当同一负载比较。

准备原始版本的独立 checkout 后，在新项目目录运行：

```bash
python bench_matrix.py --model /path/to/Qwen3-0.6B \
  --original-repo /path/to/nano-vllm-baseline --trace results/arrivals.json \
  --budgets 512 1024 2048 --repeats 3 --kv-blocks 256 --max-num-seqs 32 \
  --output-dir results/mixed_matrix
```

对 offline、injection 轨迹分别重复此命令；用不同 output-dir 保存。
`--eager` 在所有模式关闭 Graph，缺省开启 Graph。`--dry-run` 仅打印命令。
每个 mode/repeat 都在新进程中运行，轮换执行顺序；每轮保存 JSON、汇总 CSV 和逐 token CSV，
最终 aggregate.json/csv 提供均值、标准差、范围，保留全部样本。
本机已经运行轨迹生成和 matrix dry-run，未运行这些 GPU 评测。

单独运行新模式：

```bash
python bench_mixed.py --model /path/to/Qwen3-0.6B --trace results/arrivals.json \
  --mode mixed --budget 512 --kv-blocks 256 --output results/mixed_512.json
```

公平性和指标口径：

- 相同 token 轨迹、max_tokens、模型精度、Graph 设置、KV 物理块数、缓存冷热策略。
  warmup 不计时；默认 cold 清空 Prefix Cache 元数据，不重置已经预热的算子和 Graph。
  warm 保留相同预热负载后的缓存，具体命中也可能受到策略自身的驱逐顺序影响。
- 原版通过只在基准进程中替换池初始化固定相同容量，并在 postprocess 周围观测
  真正新增 token；不改原版调度/模型代码。基准默认 TP=1，TP=2 独立做正确性 smoke。
- 记录计划到达、实际入队，step 间按 `perf_counter` 注入，不按 step 数注入。
  TTFT 从实际入队起算；计划到达延误保存在原始记录中。
- 时间戳在 rank 0 取得 CPU token 后采集。所有输出指标均为 **Engine 侧**，不含 HTTP。
- 平均 TPOT 是每请求平均 ITL 的非加权平均；P50/P95/P99 ITL 将所有请求的相邻输出
  token 间隔合并后计算。只有一个输出的请求不贡献 ITL，P99 不是 P99 TPOT。
- 输出吞吐分子是实际新增 completion token；部分 Prefill 和重算不贡献输出。
  分别记录 Prefill/Decode 计算量、重算、抢占、每步 KV 占用及逐 token 时间戳。
- 汇总包含模型配置指纹/权重文件大小、dtype、库版本、GPU、配置、Git revision、
  dirty 状态和 Python 源码指纹。matrix 会拒绝合并轨迹或环境不一致的结果。
- 512/1024/2048 预算会改变 TTFT、ITL 和吞吐取舍；应报告各完整测试的波动和回归，
  不使用此前对话中的假设数字或原 README 历史性能表作为此次结果。

## 文件和后续接入点

| 文件 | 用途 |
|---|---|
| engine/scheduler.py | Decode 优先、chunk 分配、抢占、按 ID 提交 |
| engine/scheduler_output.py | 冻结的调度快照、CPU 元数据、Graph bucket |
| engine/sequence.py | 计算高水位、完整序列化 |
| engine/block_manager.py | 按缺少的实际块数检查/追加 Decode 槽位 |
| engine/model_runner.py | 统一 prepare_batch、一次 forward、warmup/Graph/TP |
| utils/context.py、layers/attention.py | 混批有效 KV、query 边界和 paged varlen |
| layers/embed_head.py | 显式采样行选择 |
| engine/metrics.py、engine/llm_engine.py | StepStats、CPU 可用时间戳、generate 兼容 |
| config.py、sampling_params.py、__init__.py | 配置/输入校验、CPU 测试可独立导入 |
| tests/、bench_mixed.py、bench_matrix.py、tools/ | 正确性测试、固定轨迹、重复评测与示例 |

若以后实现 Prompt KV 增量分配，可在 `BlockManager.can_allocate/allocate` 与
Scheduler 的 Prefill 选取处接入“仅为 end_pos 保留页”；统一输入映射已经只访问
本轮 `[start,end)`，无需改变 Attention 和采样协议。本次保留原有整段分配策略。

## 后端接口依据

本地无 FlashAttention 可检查签名，因此增加启动时的实际签名检查，并参考
[FlashAttention 2.7.4 接口源码](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/flash_attn/flash_attn_interface.py)
中的 varlen `block_table` 参数及
[官方因果掩码说明](https://github.com/Dao-AILab/flash-attention#21-change-behavior-of-causal-flag)。
依赖限定 `flash-attn>=2.5,<3`，目标环境仍需执行上述 GPU 验证。
