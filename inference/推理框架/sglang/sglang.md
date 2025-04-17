- sglang 原理
- [x] https://zhuanlan.zhihu.com/p/28199298728
- sglang对deepseek的优化
- [x] https://docs.sglang.ai/references/deepseek.html
- sglang支持[request](https://docs.sglang.ai/backend/send_request.html)
- [x] 支持文本生成
- [x] 支持视觉语言模型(Vision Language Models,embedding,图文等)
- [x] 支持奖励模型(Reward Models)
- 访问
```
curl -s http://10.236.13.81:35519/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{{"model": "/var/lib/data/aicache/models/system/DeepSeek-R1-Distill-Qwen-7B", "messages": [{{"role": "user", "content": "What is the capital of France?"}}]}}'
```
# 启动参数
```
server_args=ServerArgs(model_path='/var/lib/data/aicache/models/system/DeepSeek-R1-Distill-Qwen-7B', tokenizer_path='/var/lib/data/aicache/models/system/DeepSeek-R1-Distill-Qwen-7B', tokenizer_mode='auto', load_format='auto', trust_remote_code=False, dtype='auto', kv_cache_dtype='auto', quantization_param_path=None, quantization=None, context_length=None, device='cuda', served_model_name='/var/lib/data/aicache/models/system/DeepSeek-R1-Distill-Qwen-7B', chat_template=None, is_embedding=False, revision=None, skip_tokenizer_init=False, host='0.0.0.0', port=30000, mem_fraction_static=0.88, max_running_requests=None, max_total_tokens=None, chunked_prefill_size=8192, max_prefill_tokens=16384, schedule_policy='lpm', schedule_conservativeness=1.0, cpu_offload_gb=0, prefill_only_one_req=False, tp_size=1, stream_interval=1, stream_output=False, random_seed=464949377, constrained_json_whitespace_pattern=None, watchdog_timeout=300, download_dir=None, base_gpu_id=0, log_level='info', log_level_http=None, log_requests=False, show_time_cost=False, enable_metrics=False, decode_log_interval=40, api_key=None, file_storage_pth='sglang_storage', enable_cache_report=False, dp_size=1, load_balance_method='round_robin', ep_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', lora_paths=None, max_loras_per_batch=8, lora_backend='triton', attention_backend='flashinfer', sampling_backend='flashinfer', grammar_backend='outlines', speculative_draft_model_path=None, speculative_algorithm=None, speculative_num_steps=5, speculative_num_draft_tokens=64, speculative_eagle_topk=8, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, disable_radix_cache=False, disable_jump_forward=False, disable_cuda_graph=False, disable_cuda_graph_padding=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, disable_mla=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_ep_moe=False, enable_torch_compile=False, torch_compile_max_bs=32, cuda_graph_max_bs=160, cuda_graph_bs=None, torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, allow_auto_truncate=False, enable_custom_logit_processor=False, tool_call_parser=None, enable_hierarchical_cache=False)
```
```
[--tokenizer-path TOKENIZER_PATH] [--host HOST]
	[--port PORT] [--tokenizer-mode {auto,slow}]
	[--skip-tokenizer-init]
	[--load-format {auto,pt,safetensors,npcache,dummy,gguf,bitsandbytes,layered}]
	[--trust-remote-code]
	[--dtype {auto,half,float16,bfloat16,float,float32}]
	[--kv-cache-dtype {auto,fp8_e5m2,fp8_e4m3}]
	[--quantization-param-path QUANTIZATION_PARAM_PATH]
	[--quantization {awq,fp8,gptq,marlin,gptq_marlin,awq_marlin,bitsandbytes,gguf,modelopt,w8a8_int8}]
	[--context-length CONTEXT_LENGTH]
	[--device {cuda,xpu,hpu,cpu}]
	[--served-model-name SERVED_MODEL_NAME]
	[--chat-template CHAT_TEMPLATE] [--is-embedding]
	[--revision REVISION]
	[--mem-fraction-static MEM_FRACTION_STATIC]
	[--max-running-requests MAX_RUNNING_REQUESTS]
	[--max-total-tokens MAX_TOTAL_TOKENS]
	[--chunked-prefill-size CHUNKED_PREFILL_SIZE]
	[--max-prefill-tokens MAX_PREFILL_TOKENS]
	[--schedule-policy {lpm,random,fcfs,dfs-weight}]
	[--schedule-conservativeness SCHEDULE_CONSERVATIVENESS]
	[--cpu-offload-gb CPU_OFFLOAD_GB]
	[--prefill-only-one-req PREFILL_ONLY_ONE_REQ]
	[--tensor-parallel-size TENSOR_PARALLEL_SIZE]
	[--stream-interval STREAM_INTERVAL] [--stream-output]
	[--random-seed RANDOM_SEED]
	[--constrained-json-whitespace-pattern CONSTRAINED_JSON_WHITESPACE_PATTERN]
	[--watchdog-timeout WATCHDOG_TIMEOUT]
	[--download-dir DOWNLOAD_DIR]
	[--base-gpu-id BASE_GPU_ID] [--log-level LOG_LEVEL]
	[--log-level-http LOG_LEVEL_HTTP] [--log-requests]
	[--show-time-cost] [--enable-metrics]
	[--decode-log-interval DECODE_LOG_INTERVAL]
	[--api-key API_KEY]
	[--file-storage-pth FILE_STORAGE_PTH]
	[--enable-cache-report]
	[--data-parallel-size DATA_PARALLEL_SIZE]
	[--load-balance-method {round_robin,shortest_queue}]
	[--expert-parallel-size EXPERT_PARALLEL_SIZE]
	[--dist-init-addr DIST_INIT_ADDR] [--nnodes NNODES]
	[--node-rank NODE_RANK]
	[--json-model-override-args JSON_MODEL_OVERRIDE_ARGS]
	[--lora-paths [LORA_PATHS ...]]
	[--max-loras-per-batch MAX_LORAS_PER_BATCH]
	[--lora-backend LORA_BACKEND]
	[--attention-backend {flashinfer,triton,torch_native}]
	[--sampling-backend {flashinfer,pytorch}]
	[--grammar-backend {xgrammar,outlines}]
	[--speculative-algorithm {EAGLE}]
	[--speculative-draft-model-path SPECULATIVE_DRAFT_MODEL_PATH]
	[--speculative-num-steps SPECULATIVE_NUM_STEPS]
	[--speculative-num-draft-tokens SPECULATIVE_NUM_DRAFT_TOKENS]
	[--speculative-eagle-topk {1,2,4,8}]
	[--enable-double-sparsity]
	[--ds-channel-config-path DS_CHANNEL_CONFIG_PATH]
	[--ds-heavy-channel-num DS_HEAVY_CHANNEL_NUM]
	[--ds-heavy-token-num DS_HEAVY_TOKEN_NUM]
	[--ds-heavy-channel-type DS_HEAVY_CHANNEL_TYPE]
	[--ds-sparse-decode-threshold DS_SPARSE_DECODE_THRESHOLD]
	[--disable-radix-cache] [--disable-jump-forward]
	[--disable-cuda-graph] [--disable-cuda-graph-padding]
	[--disable-outlines-disk-cache]
	[--disable-custom-all-reduce] [--disable-mla]
	[--disable-overlap-schedule] [--enable-mixed-chunk]
	[--enable-dp-attention] [--enable-ep-moe]
	[--enable-torch-compile]
	[--torch-compile-max-bs TORCH_COMPILE_MAX_BS]
	[--cuda-graph-max-bs CUDA_GRAPH_MAX_BS]
	[--cuda-graph-bs CUDA_GRAPH_BS [CUDA_GRAPH_BS ...]]
	[--torchao-config TORCHAO_CONFIG]
	[--enable-nan-detection] [--enable-p2p-check]
	[--triton-attention-reduce-in-fp32]
	[--triton-attention-num-kv-splits TRITON_ATTENTION_NUM_KV_SPLITS]
	[--num-continuous-decode-steps NUM_CONTINUOUS_DECODE_STEPS]
	[--delete-ckpt-after-loading] [--enable-memory-saver]
	[--allow-auto-truncate]
	[--enable-custom-logit-processor]
	[--tool-call-parser {qwen25,mistral,llama3}]
	[--enable-hierarchical-cache]

```
# 聊天参数
```
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[ ## 多轮对话
	
        {
            "role": "system", #system 角色设定了助手的身份是"一个知识渊博的历史学家，回答要简洁"
            "content": "You are a knowledgeable historian who provides concise responses.",
        },
        {"role": "user", "content": "Tell me about ancient Rome"},
        {
            "role": "assistant", #assistant 是前一次模型生成的回答。
            "content": "Ancient Rome was a civilization centered in Italy.",
        },
        {"role": "user", "content": "What were their major achievements?"},
    ],
    temperature=0.3,  # Lower temperature for more focused responses控制生成的多样性，值越低越保守
    max_tokens=128,  # Reasonable length for a concise response限制输出最大 token 数（最多生成 128 个 token）。
    top_p=0.95,  # Slightly higher for better fluency；nucleus sampling（控制输出词汇的多样性），设为 0.95 表示只考虑累计概率前 95% 的词。
    presence_penalty=0.2,  # Mild penalty to avoid repetition #轻微惩罚重复出现的主题，鼓励模型引入新内容。
    frequency_penalty=0.2,  # Mild penalty for more natural language，防止模型重复相同的词。
    n=1,  # Single response is usually more stable只生成 1 个回答。
    seed=42,  # Keep for reproducibility: 使用固定随机种子以实现结果可复现。
	 stream=True, #流shi
	stop=None, 模型可以自由生成，直到自然结束或达上限;stop=["\n"]模型遇到换行符就停止生成;stop=["User:", "Assistant:"]	模型遇到这些角色提示词就停止
	
)

print_highlight(response.choices[0].message.content)
```
# 将模型转为特定格式
```
python convert.py --hf-ckpt-path /path/to/DeepSeek-V3 --save-path /path/to/DeepSeek-V3-Demo --n-experts 256 --model-parallel 16
```
# 参数
## 常用
- xxx
1.  --enable-torch-compile：则第一次编译模型torch.compile需要一些时间。你可以参考这里优化编译结果的缓存，这样就可以利用缓存来加快下次启动的速度。
2. --enable-dp-attention：启用 Distributed Parallel Attention（分布式并行注意力机制） 的参数。
启用数据并行关注后，与以前的版本相比，我们的解码吞吐量提高了 1.9 倍。此优化旨在提高吞吐量，应用于 QPS（每秒查询数）较高的场景。它可以由 --enable-dp-attention for DeepSeek 模型启用。
![dp](https://paddlepedia.readthedocs.io/en/latest/_images/xor.png){: style="width:90%"}
从 v0.4.4 开始，DP 和 TP 注意力可以灵活组合。例如，要在 '8*H100' 的 2 个节点上部署 DeepSeek-V3/R1，您可以指定 --tp 16 和 --dp 2 ，这意味着对于注意力部分有 2 个 DP 组，每个 DP 组中有 8 个 TP 组
3. --attention-backend: 配置MLA 注意力后端，包括FlashAttention3、Flashinfer 和 Triton 后端
4. --disable-mla: 关闭mla 
5. SGL_ENABLE_JIT_DEEPGEMM=1 : 开启deepseek的DeepGEMM
6. --speculative-algorithm 、 --speculative-draft-model-path 、 --speculative-num-steps --speculative-eagle-topk 和 --speculative-num-draft-tokens： SGLang 基于 EAGLE 推测解码实现 DeepSeek V3 多令牌预测 （MTP）。通过此优化，在 H200 TP8 设置下，批量大小 1 的解码速度可以分别提高 1.8 倍，批量大小 32 的解码速度可以分别提高 1.5 倍。
```
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3-0324 --speculative-algorithm EAGLE --speculative-draft-model-path lmsys/DeepSeek-V3-0324-NextN --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 --trust-remote-code --tp 8
```
7. --dist-timeout 3600 :模型加载时间延长，这允许 1 小时的超时。
8. --context-length: **防止出现OOM**，调整--context-length以避免 GPU 内存不足问题。对于 Scout 型号，我们建议在 8*H100 上将此值设置为最大 1M，在 8*H200 上将此值设置为最大 2.5M。对于 Maverick 模型，我们不需要在 8*H200 上设置上下文长度。
9. --chat-template： 聊天模板，例如：--chat-template openai ，--chat-template  llama-4
10. --tp 2 :要启用多 GPU 张量并行，请添加 --tp 2 .如果报错 “not supported peer access between these two devices”，请在服务器启动命令中添加--enable-p2p-check。
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 2

```
11. --dp 2 :要启用多 GPU 数据并行性，请添加 --dp 2 .如果有足够的内存，数据并行性对吞吐量更好。它还可以与张量并行一起使用。以下命令总共使用 4 个 GPU。我们建议使用 SGLang Router 实现数据并行。
```
python -m sglang_router.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --dp 2 --tp 2
```
12. --mem-fraction-static: 如果您在提供服务期间看到内存不足错误，请尝试通过设置较小的值 --mem-fraction-static .默认值为 0.9 .
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --mem-fraction-static 0.7
```
13.--shm-size: 对于 docker 和 Kubernetes 运行，您需要设置用于进程之间通信的共享内存。请参阅 --shm-size Kubernetes 清单的 docker 和/dev/shm大小更新。
14.--chunked-prefill-size: 如果在预填充长提示期间看到内存不足错误，请尝试设置较小的分块预填充大小。
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --chunked-prefill-size 4096
```
15. --enable-torch-compile :启用torch.compile加速，可加速小批量的小模型。默认情况下，缓存路径位于 /tmp/torchinductor_root,可以使用 环境变量 TORCHINDUCTOR_CACHE_DIR 对其进行自定义。更多细节请参考 PyTorch 官方文档 和 为 torch.compile 启用缓存 。
16. --torchao-config int4wo-128，启用 torchao 量化
17. --quantization fp8，启用 fp8 权重量化
18.	--kv-cache-dtype fp8_e5m2，启用 fp8 kv 缓存量化
19. --nnodes 2，要在多个节点上运行张量并行，请添加 --nnodes 2 .如果您有两个节点，每个节点上有两个 GPU，并且想要运行TP=4，设sgl-dev-0为第一个节点的主机名并50000成为可用端口，则可以使用以下命令。如果遇到死锁，请尝试添加 --disable-cuda-graph。
20. load_format ，加载权重的格式。默认为 *.safetensors /*.bin
21. dtype ，用于模型的 Dtype，默认为 bfloat16
22. kv_cache_dtype ，kv 缓存的 Dtype 类型，默认为 dtype
23. revision ，调整是否应使用模型的特定版本。
24. skip_tokenizer_init，设置为 true 可向引擎提供令牌并直接获取输出令牌，
25. json_model_override_args，  使用提供的 JSON 覆盖模型配置
26. delete_ckpt_after_loading， 加载模型后删除模型 checkpoint。
27. disable_fast_image_processor 采用 base 图像处理器而不是 fast image processor（默认）
28. load_balance_method ：将被弃用。数据并行请求的负载均衡策略。
29. enable_ep_moe： 启用专家并行性，将专家分配到 MoE 模型的多个 GPU 上。
30. ep_size EP 的大小。请使用 ，将模型权重分片tp_size=ep_size，有关详细的基准测试，请参阅此 PR 。如果未设置，ep_size则将自动设置为 tp_size 。
31. enable_deepep_moe 启用专家并行性，将专家分配到基于 deepseek-ai/DeepEP 的 DeepSeek-V3 模型的多个 GPU 上。 
32. deepep_mode 选择启用 DeepEP MoE 时的模式，可以是 normal ，low_latency或 auto 。默认值为 auto ，表示 low_latency decode batch 和 normal prefill batch。
32. m_fraction_static ：用于静态内存（如模型权重和 KV 缓存）的可用 GPU 内存的分数。如果构建 KV 缓存失败，则应增加它。如果 CUDA 内存不足，则应减少内存。
33. max_running_requests ：并发运行的最大请求数。
34. max_total_tokens ：可以存储到 KV 缓存中的最大 Token 数。主要用于调试。
35. chunked_prefill_size ：以这些大小的块执行预填充。较大的块大小会加快预填充阶段，但会增加 VRAM 消耗。如果 CUDA 内存不足，则应减少内存。
36. max_prefill_tokens ：一个预填充批次中要接受的令牌数量的令牌预算。实际数字是此参数和 context_length .
37. schedule_policy ：用于控制单个引擎中等待预填充请求的处理顺序的计划策略。
38. schedule_conservativeness ：可用于在接受新请求时降低/增加服务器的保守性。高度保守的行为会导致饥饿，但低保守会导致性能变慢。
39. cpu_offload_gb ：保留此 RAM 量（以 GB 为单位），用于将模型参数卸载到 CPU
40. stream_interval ：流式响应的间隔（以令牌为单位）。较小的值会导致更平滑的流式处理，而较大的值会带来更好的吞吐量。
41. random_seed ：可用于强制实施更具确定性的行为
42. watchdog_timeout ：如果批处理生成时间过长，则在终止服务器之前调整看门狗线程的超时。
43. download_dir ：用于覆盖模型权重的默认 Hugging Face 缓存目录。
44. base_gpu_id ：用于调整用于在可用 GPU 之间分配模型的第一个 GPU。
45. allow_auto_truncate ：自动截断超过最大输入长度的请求。
46. tp_size ：张量并行。模型权重分片的 GPU 数量。主要是为了节省内存而不是为了高吞吐量，[博客文章](https://pytorch.org/tutorials/intermediate/TP_tutorial.html#how-tensor-parallel-works/)
47. dp_size:数据并行，**dp_size ：将被弃用**。模型的数据并行副本数。 **建议使用 SGLang 路由器，而不是当前的 naive 数据并行**。
## 日志
- xxx
1. log_level 全局日志详细程度。
2. log_level_http ：HTTP 服务器日志的单独详细级别（如果未设置，则默认为 log_level ）。
3. log_requests ：记录所有调试请求的输入和输出。
4. log_requests_level ：范围从 0 到 2：级别 0 仅显示请求中的一些基本元数据，级别 1 和 2 显示请求详细信息（例如，文本、图像），级别 1 将输出限制为 2048 个字符（如果未设置，则默认为 0 ）。
5. show_time_cost ：打印或记录内部作的详细时序信息（有助于性能调整）
6. enable_metrics ：导出请求使用情况和性能的类似 Prometheus 的指标。
7. decode_log_interval ：记录解码进度的频率（以令牌为单位）。
## 多节点分布式服务
- xxx
1. dist_init_addr ：用于初始化 PyTorch 分布式后端的 TCP 地址（例如 192.168.0.2:25000 ）
2. nnodes ：集群中的节点总数。
3. node_rank ：此节点在分布式设置nnodes中的排名
```
# Node 0
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 4 --dist-init-addr sgl-dev-0:50000 --nnodes 2 --node-rank 0

# Node 1
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 4 --dist-init-addr sgl-dev-0:50000 --nnodes 2 --node-rank 1
```
## LoRA 系列
- xxx
1. lora_paths 您可以将适配器列表作为列表提供给您的模型。每个 batch 元素都将获得应用了相应 lora 适配器的模型响应。当前 cuda_graph 和 radix_attention 不支持此选项，因此您需要手动禁用它们。我们仍在努力解决这些问题。
2. max_loras_per_batch ：正在运行的批处理（包括基础模型）中的最大 LoRA 数。
3. lora_backend ：为 Lora 模块运行 GEMM 内核的后端可以是 triton 或 flashinfer 之一。默认为 triton 。
## Kernel backend
- xxx
1. attention_backend ：此参数指定注意力计算和 KV 缓存管理的后端，可以是 fa3、flashinfer triton 、torch_native 、。部署 DeepSeek 模型时，请使用此参数指定 MLA 后端。
2. sampling_backend ：采样的后端。
## Constrained  Decoding（受限解码）
1. grammar_backend ：用于约束解码的语法后端。
2. constrained_json_whitespace_pattern ：与Outlines语法后端一起使用，以允许 JSON 带有同义换行符、制表符或多个空格。详情可在此处找到。
## Speculative decoding（推测解码）
1. speculative_draft_model_path ：用于推测解码的 draft 模型路径。
2. speculative_algorithm ：推测解码的算法。目前仅支持 Eagle。请注意，在使用 eagle 推测解码时，基数缓存、分块预填充和重叠调度程序将被禁用。
3. speculative_num_steps ：在验证之前我们运行了多少次草稿。
4. speculative_num_draft_tokens ：草稿中提议的代币数量。
5. speculative_eagle_topk ：我们在 Eagle 的每个步骤中保留以供验证的顶级候选项的数量。
6. speculative_token_map ： 可选，FR-Spec 的高频令牌列表的路径，用于加速 Eagle 。
## Double Sparsity 双稀疏度# 
1. enable_double_sparsity ：启用双倍稀疏性，从而提高吞吐量
2. ds_channel_config_path ：双倍稀疏配置。
3. ds_heavy_channel_num ：每层要保留的通道索引数。
4. ds_heavy_token_num ：解码过程中用于注意力的标记数。如果min_seq_len批量<此数字，则跳过稀疏解码。
5. ds_heavy_channel_type ：重通道的类型。或 q qk 或 k .
6. ds_sparse_decode_threshold ：如果max_seq_len批量<此阈值，则不应用稀疏解码。

## Debug options 调试选项# 
- xxx
1. disable_radix_cache ：禁用 Radix 后端以进行前缀缓存。
2. disable_cuda_graph ：禁用 cuda 图进行模型转发。如果遇到无法纠正的 CUDA ECC 错误，请使用此错误。
3. disable_cuda_graph_padding ：当需要填充时禁用 cuda 图。在其他情况下，仍然使用 cuda 图。
4. disable_outlines_disk_cache ：禁用大纲语法后端的磁盘缓存
5. disable_custom_all_reduce ：禁用自定义 all reduce 内核。
6. disable_mla ：禁用 Deepseek 模型的多头潜在注意。
7. disable_overlap_schedule ：禁用 Overhead-Scheduler 。
8. enable_nan_detection ：启用此选项后，如果 logit 包含 NaN ，则采样器会打印警告。
9. enable_p2p_check ：关闭在访问 GPU 时始终允许 p2p 检查的默认值。
10. triton_attention_reduce_in_fp32 ：在 triton 核中，这会将中间注意力结果投射到 float32 。

## Optimization  优化# 
- 其中一些选项仍处于实验阶段。
1. enable_mixed_chunk ：启用混合预填充和解码，[请参阅此讨论](https://github.com/sgl-project/sglang/discussions/1163/)
2. enable_dp_attention ：为 Deepseek 模型启用数据并行 Attention。
3. enable_torch_compile ：Torch 编译模型。请注意，编译模型需要很长时间，但性能会得到很大提升。编译后的模型也可以缓存以备将来使用。
4. torch_compile_max_bs ：使用 torch_compile 时的最大批量大小。
5. cuda_graph_max_bs ：调整使用 cuda 图时的最大 batchsize。默认情况下，这是根据 GPU 具体情况为您选择的。
6. cuda_graph_bs ：要捕获的CudaGraphRunner批大小。
7. torchao_config ：使用 torchao 优化模型的实验性功能。可能的选项包括：int8dq、int8wo、int4wo-<group_size>、fp8wo、fp8dq-per_tensor、fp8dq-per_row。
8. triton_attention_num_kv_splits ：用于调整 triton 内核中 KV 拆分的数量。默认值为 8。
9. enable_flashinfer_mla ：将注意力后端与 FlashInfer MLA 包装器一起用于 DeepSeek 模型。 **此参数将在下一版本中弃用**。请--attention_backend flashinfer改用 来启用 FlashfIner MLA。
10. flashinfer_mla_disable_ragged ：禁止对 FlashInfer MLA 注意力后端使用参差不齐的预填充包装器。仅在 FlashInfer 用作 MLA 后端时使用它。

## 采样参数
 
[未翻译](https://docs.sglang.ai/backend/sampling_params.html/)

## Hyperparameter Tuning 超参数调优
实现大批量是实现高吞吐量的最重要因素，服务器满载运行时，请在日志中查找以下内容：
```
Decode batch. #running-req: 233, #token: 370959, token usage: 0.82, gen throughput (token/s): 4594.01, #queue-req: 317
```
###  Tune Your Request Submission Speed 调整提交速度
queue-req表示队列中的请求数。如果您经常看到 #queue-req == 0 ，则表明您受到请求提交速度的瓶颈。正常范围#queue-req是 50 - 500 。
另一方面，不要做得#queue-req太大，因为它也会增加服务器上的调度开销，尤其是在使用默认的 longest-prefix-match 调度策略 （--schedule-policy lpm ） 时。
## --schedule-conservativeness  调整 --schedule-conservativeness
- token usage表示服务器的 KV 缓存内存利用率。token usage > 0.9表示良好的利用率。如果您经常看到 token usage < 0.9 和 #queue-req > 0 ，则表示服务器在接收新请求方面过于保守。您可以减小--schedule-conservativeness到类似于 0.3 的值。当用户发送许多请求时，可能会发生服务器过于保守的情况，其中包含大量max_new_tokens请求，但由于 EOS 或 stop 字符串，请求会提前停止。
- 另一方面，如果您看到token usage非常高并且经常看到类似 decode out of memory happened, #retracted_reqs: 1, #new_token_ratio: 0.9998 -> 1.0000 的警告，则可以增加到 --schedule-conservativeness 1.3 之类的值。如果您偶尔看到但不经常看到decode out of memory happened，那没关系。
### --dp-size 和--tp-size
数据并行性更有利于吞吐量。当有足够的 GPU 内存时，始终支持数据并行性以提高吞吐量。请参阅 sglang router 以获得更好的数据并行性，而不是使用 dp_size parameter。

### 通过优化 --chunked-prefill-size 、 --mem-fraction-static --max-running-requests 、 避免内存不足
如果您看到内存不足 （OOM） 错误，可以尝试优化以下参数。
- 如果在预填充期间发生 OOM，请尝试将 OOM 减少--chunked-prefill-size到 4096 或 2048
- 如果在解码过程中发生 OOM，请尝试减少 --max-running-requests 。
- 您还可以尝试 decrease --mem-fraction-static ，这减少了 KV 高速缓存内存池的内存使用量，并有助于预填充和解码。

### 为 启用缓存torch.compile
- 要启用torch.compile加速，请添加 --enable-torch-compile .**它可加速小批量的小模型**。默认情况下，torch.compile将自动缓存 FX 图和 Triton 在 /tmp/torchinductor_root 中，这可能会根据系统策略 清除。您可以导出环境变量TORCHINDUCTOR_CACHE_DIR以将编译缓存保存在所需的目录中，以避免不必要的删除。您还可以与其他计算机共享缓存以减少编译时间。
- SGLang 使用 max-autotune-no-cudagraphs 的模式为 torch.compile 。自动调整可能会很慢。如果要在许多不同的计算机上部署模型，可以将torch.compile缓存发送到这些计算机并跳过编译步骤。这是基于 PyTorch 官方文档的。
```
 例子:通过设置TORCHINDUCTOR_CACHE_DIR并运行模型一次来生成缓存。
 TORCHINDUCTOR_CACHE_DIR=/root/inductor_root_cache python3 -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --enable-torch-compile
 
 将缓存文件夹复制到其他计算机，然后使用 TORCHINDUCTOR_CACHE_DIR 启动服务器。
 
```
### 调整 --schedule-policy
- 如果工作负载有许多共享前缀，请使用默认 --schedule-policy lpm .其中lpm代表最长的前缀匹配。
- 如果您根本没有共享前缀，或者始终将带有共享前缀的请求一起发送，则可以尝试 --schedule-policy fcfs 。其中fcfs代表 先到先得。此策略具有较低的计划开销。



# SGLang 原生 API
- [x] /generate (text generation model) /generate （文本生成模型） 
- [x] /get_model_info
1. model_path ：模型的路径/名称
2. is_generation ：模型是用作生成模型还是嵌入模型。
3. tokenizer_path ：分词器的路径名称。
- [x] /get_server_info
```
{"model_path"："meta-llama/llama-3.2-1B-Instruct"，"tokenizer_path"："meta-llama/llama-3.2-1B-Instruct"，"tokenizer_mode"："auto"，"skip_tokenizer_init"：false，"load_format"："auto"，"trust_remote_code"：false，"dtype"："auto"，"kv_cache_dtype"："auto"，"quantization"：null，"quantization_param_path"：null，"context_length"：null，"device"："cuda"，"served_model_name"："meta-llama/llama-3.2-1B-Instruct"，"chat_template"：null，"completion_template"：null，"is_embedding"：false，"revision"：null，"host"："0.0.0.0"，"port"：36326，"mem_fraction_static"：0.88，"max_running_requests"：200，"max_total_tokens"：20480，"chunked_prefill_size"：null，"max_prefill_tokens"：16384，"schedule_policy"："fcfs"，"schedule_conservativeness"：1.0，"cpu_offload_gb"：0，"page_size"：1，"tp_size"：1，"stream_interval"：1，"stream_output"：false，"random_seed"：19128996，"constrained_json_whitespace_pattern"：null，"watchdog_timeout"：300，"dist_timeout"：null，"download_dir"：null，"base_gpu_id"：0，"gpu_id_step"：1，"log_level"："info"，"log_level_http"：null，"log_requests"：false，"log_requests_level"：0，"show_time_cost"：false，"enable_metrics"：false，"decode_log_interval"：40，"api_key"：null，"file_storage_path"："sglang_storage"，"enable_cache_report"：false，"reasoning_parser"：null，"dp_size"：1，"load_balance_method"："round_robin"，"ep_size"：1，"dist_init_addr"：null，"nnodes"：1，"node_rank"：0，"json_model_override_args"："{}"，"lora_paths"：null，"max_loras_per_batch"：8，"lora_backend"："triton"，"attention_backend"："flashinfer"，"sampling_backend"："flashinfer"，"grammar_backend"："xgrammar"，"speculative_algorithm"：null，"speculative_draft_model_path"：null，"speculative_num_steps"：null，"speculative_eagle_topk"：null，"speculative_num_draft_tokens"：null，"speculative_accept_threshold_single"：1.0，"speculative_accept_threshold_acc"：1.0，"speculative_token_map"：null，"enable_double_sparsity"：false，"ds_channel_config_path"：null，"ds_heavy_channel_num"：32，"ds_heavy_token_num"：256，"ds_heavy_channel_type"："qk"，"ds_sparse_decode_threshold"：4096，"disable_radix_cache"：false，"disable_cuda_graph"：true，"disable_cuda_graph_padding"：false，"enable_nccl_nvls"：false，"disable_outlines_disk_cache"：false，"disable_custom_all_reduce"：false，"disable_mla"：false，"enable_llama4_multimodal"：null，"disable_overlap_schedule"：false，"enable_mixed_chunk"：false，"enable_dp_attention"：false，"enable_ep_moe"：false，"enable_deepep_moe"：false，"deepep_mode"："auto"，"enable_torch_compile"：false，"torch_compile_max_bs"：32，"cuda_graph_max_bs"：160，"cuda_graph_bs"：null，"torchao_config"：""，"enable_nan_detection"：false，"enable_p2p_check"：false，"triton_attention_reduce_in_fp32"：false，"triton_attention_num_kv_splits"：8，"num_continuous_decode_steps"：1，"delete_ckpt_after_loading"：false，"enable_memory_saver"：false，"allow_auto_truncate"：false，"enable_custom_logit_processor"：false，"tool_call_parser"：null，"enable_hierarchical_cache"：false，"hicache_ratio"：2.0，"enable_flashinfer_mla"：false，"enable_flashmla"：false，"flashinfer_mla_disable_ragged"：false，"warmups"：null，"n_share_experts_fusion"：0，"disable_shared_experts_fusion"：false，"debug_tensor_dump_output_folder"：null，"debug_tensor_dump_input_file"：null，"debug_tensor_dump_inject"：false，"disaggregation_mode"："null"，"disaggregation_bootstrap_port"：8998，"disaggregation_transfer_backend"："mooncake"，"disable_fast_image_processor"：false，"status"："ready"，"max_total_num_tokens"：20480，"max_req_input_len"：20474，"use_mla_backend"：false，"last_gen_throughput"：133.92117295460932，"version"："0.4.5"}
```
- [x] /health :检查服务器的运行状况。
- [x] /health_generate  ：通过生成一个令牌来检查服务器的运行状况。
- [x] /flush_cache ：刷新 radix cache。当 /update_weights API 更新模型权重时，它将自动触发。
- [x] /update_weights 
- [x] /encode(embedding model)
- [x] /classify(reward奖励 model)
- [x] /start_expert_distribution_record
- [x] /stop_expert_distribution_record	
- [x] /dump_expert_distribution_record
- [x] /update_weights_from_disk 
在不重新启动服务器的情况下从磁盘更新模型权重。仅适用于架构和参数大小相同的模型。SGLang 支持update_weights_from_disk用于训练期间持续评估的 API（将检查点保存到磁盘并从磁盘更新权重）。

- 推理模式
1.非流式同步生成
2.流式同步生成
3.非流式异步生成
4.流式异步生成

# 结构化输出
- 当使用使用特殊标记（如<think>...</think>表示推理部分）的推理模型时，您可能希望在这些部分中允许自由格式的文本，同时仍然对输出的其余部分强制执行语法约束。
- SGLang 提供了一项功能，可以在推理部分内禁用语法限制。这对于需要在提供结构化输出之前执行复杂推理步骤的模型特别有用。
- 要启用此功能，请在启动服务器时使用--reasoning-parser来让推理引擎是否输出think_end_token，例如 </think> 。您还可以使用 --reasoning-parser flag 指定推理解析器。
- 支持的型号
- [x] DeepSeek R1 系列 ： 推理内容用 <think> and </think> 标签包装。
- [x] QwQ ： 推理内容用 <think> and </think> 标签包装。
例如
``` python
import openai
import os
from sglang.test.test_utils import is_in_ci

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

from sglang.utils import wait_for_server, print_highlight, terminate_process

os.environ["TOKENIZERS_PARALLELISM"] = "false"


server_process, port = launch_server_cmd(
    "python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --host 0.0.0.0 --reasoning-parser deepseek-r1"
)

wait_for_server(f"http://localhost:{port}")
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

```

```json
import json

json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }
)

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    messages=[
        {
            "role": "user",
            "content": "Give me the information of the capital of France in the JSON format.",
        },
    ],
    temperature=0,
    max_tokens=2048,
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "foo", "schema": json.loads(json_schema)},
    },
)

print_highlight(
    f"reasoing_content: {response.choices[0].message.reasoning_content}\n\ncontent: {response.choices[0].message.content}"
)

```
# 支持的模型
##   大语言模型
- 例子
```
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-1B-Instruct \  # example HF/local path
  --host 0.0.0.0 \
  --port 30000 \
```
##  Vision Language Models 视觉语言模型
- 这些模型接受多模态输入（例如，图像和文本）并生成文本输出。它们使用视觉编码器来增强语言模型，并且需要特定的聊天模板来处理视觉提示。
- 我们需要为 VLM 指定--chat-template，因为 HuggingFace tokenizer 中提供的聊天模板仅支持文本。如果未指定视觉模型的 --chat-template ，则服务器将使用 HuggingFace 的默认模板，该模板仅支持文本，并且不会传入图像。
- 例子
```
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-11B-Vision-Instruct \  # example HF/local path
  --chat-template llama_3_vision \                        # required chat template
  --host 0.0.0.0 \
  --port 30000 \
```
## Embedding Models
- GLang 通过将高效的服务机制与其灵活的编程接口集成，为嵌入模型提供强大的支持。此集成允许简化嵌入任务的处理，从而促进更快、更准确的检索和语义搜索作。SGLang 的架构可以提高资源利用率并减少嵌入模型部署的延迟。
- 它们与 --is-embedding 一起执行，有些可能需要--trust-remote-code和/或 --chat-template.
- 例子
```
python3 -m sglang.launch_server \
  --model-path Alibaba-NLP/gme-Qwen2-VL-2B-Instruct \  # example HF/local path
  --is-embedding \
  --host 0.0.0.0 \
  --chat-template gme-qwen2-vl \                     # set chat template
  --port 30000 \
```
## Reward Models 奖励模型
- 这些模型输出标量奖励分数或分类结果，通常用于强化学习或内容审核任务。
- 它们与 --is-embedding 一起执行，有些可能需要 --trust-remote-code .
- 例子
```
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-Math-RM-72B \  # example HF/local path
  --is-embedding \
  --host 0.0.0.0 \
  --tp-size=4 \                          # set for tensor parallelism
  --port 30000 \
```
## 如何支持一个新模型
### 本文档介绍如何在 SGLang 中添加对新语言模型和视觉语言模型 （VLM） 的支持。它还介绍了如何测试新模型和注册外部实现。
- 要在 SGLang 中支持新模型，您只需在 [SGLang Models Directory](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models/)  下添加一个文件。您可以从现有模型实现中学习，并为您的模型创建一个新文件。对于大多数模型，您应该能够找到类似的模型作为起点（例如，从 Llama 开始）。另请参阅如何将模型从 [vLLM 移植到 SGLang] (https://docs.sglang.ai/supported_models/support_new_models.html#port-a-model-from-vllm-to-sglang/)

## 如何支持Vision-Language 模型
### 为了在 SGLang 中支持新的视觉语言模型 （vLM），除了标准 LLM 支持之外，还有几个关键组件：
- 将新模型注册为多模态：以 [model_config.py] (https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/model_config.py/)  为单位扩展is_multimodal_model以返回True模型。
- Process  Images ：定义一个新Processor类，该类继承此BaseProcessor处理器并将其注册为模型的专用处理器。有关详细信息，请参阅 [multimodal_processor.py] (https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/multimodal_processor.py/)
- Handle Image Tokens ：为新模型实施pad_input_ids函数。在这个功能中，提示符中的图像标记应该被展开并替换为 image-hashes，这样 SGLang 在使用时就可以识别不同的图像RadixAttention。
- Replace Vision Attention: 用 SGLang 的 ViT 替换 Multiheaded Attention VisionAttention 。

### 测试正确性
#### 交互式调试
对于交互式调试，请比较 Hugging Face/Transformers 和 SGLang 的输出。以下两个命令应提供相同的文本输出和非常相似的预填充 logits：
- Get the reference output: 获取参考输出： 
python3 scripts/playground/reference_hf.py --model-path [new model] --model-type {text,vlm}
- 获取 SGLang 输出： 
python3 -m sglang.bench_one_batch --correct --model [new model]

#### Add the Model to the Test Suite 将模型添加到测试套件#
- 为确保新模型得到良好的维护，请将其添加到测试套件中，将其包含在 test_generation_models.py 文件的ALL_OTHER_MODELS列表中，在本地计算机上测试新模型，并在 PR 中报告演示性基准测试（GSM8K、MMLU、MMMU、MMMU-Pro 等）的结果。
- 这是在本地计算机上测试新模型的命令：
```
ONLY_RUN=Qwen/Qwen2-1.5B python3 -m unittest test_generation_models.TestGenerationModels.test_others
```
## 将模型从 vLLM 移植到 SGLang
vLLM 模型目录是一项宝贵的资源，因为 vLLM 涵盖了许多模型。SGLang 重用了 vLLM 的接口和一些层，从而可以更轻松地将模型从 vLLM 移植到 SGLang,要将模型从 vLLM 移植到 SGLang.
- 比较这两个文件以获取指导：
- [x] [SGLang Llama Implementation SGLang Llama 实现] (https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama.py/) 
- [x] [vLLM Llama Implementation vLLM Llama 实现 ](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py/)
- 主要区别：
- [x] 将 vLLM Attention 替换为 RadixAttention （确保传递给 layer_id RadixAttention ）。
- [x] 将 vLLM LogitsProcessor 替换为 SGLang 的 LogitsProcessor .
- [x] 将 ViT Attention 的多头替换为 SGLang 的 VisionAttention .
- [x] 将其他 vLLM 层（例如 RMSNorm 、 SiluAndMul ）替换为 SGLang 层。
- [x] Remove Sample. 删除 Sample . 
- [x] 更改forward()函数并添加方法forward_batch()。
- [x] 在末尾添加EntryClass。
- [x] 确保新实施仅使用 SGLang 组件，而不依赖于任何 vLLM 组件。

## Registering an External Model Implementation
注册外部模型实现
- 除了上述方法之外，您还可以在启动服务器之前向 注册ModelRegistry新模型。这样，您就可以在不修改源代码的情况下集成模型。
```
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.entrypoints.http_server import launch_server

# For a single model, add it to the registry:
ModelRegistry.models[model_name] = model_class

# For multiple models, you can imitate the import_model_classes() function:
from functools import lru_cache

@lru_cache()
def import_new_model_classes():
    model_arch_name_to_cls = {}
    # Populate model_arch_name_to_cls with your new model classes.
    ...
    return model_arch_name_to_cls

ModelRegistry.models.update(import_new_model_classes())

# Launch the server with your server arguments:
launch_server(server_args)
```
通过遵循这些准则，您可以在 SGLang 中添加对新语言模型和视觉语言模型的支持，并确保它们经过全面测试并轻松集成到系统中。

# Speculative Decoding 推测解码
**SGLang 现在提供基于 EAGLE 的 （EAGLE-2/EAGLE-3） 推测解码选项。我们的实施旨在最大限度地提高速度和效率，被认为是开源 LLM 引擎中最快的引擎之一。** 注意：目前 SGLang 中的推理解码兼容基数缓存和分块预填充。

## Performance Highlights 性能亮点
[Performance Highlights 性能亮点](https://docs.sglang.ai/backend/speculative_decoding.html/)

# 结构化输出
您可以指定 JSON 架构、正则表达式或 EBNF 来约束模型输出。模型输出将保证遵循给定的约束。只能为请求指定一个约束参数 （json_schema 、 regex 或 ebnf ）。
[结构化输出](https://docs.sglang.ai/backend/structured_outputs.html/)

# Tool and Function Calling 工具和函数调用
https://docs.sglang.ai/backend/function_calling.html

# Reasoning Parser 推理解析器
https://docs.sglang.ai/backend/separate_reasoning.html

# Custom Chat Template 自定义聊天模板
https://docs.sglang.ai/backend/custom_chat_template.html

# Quantization 量化# 
https://docs.sglang.ai/backend/quantization.html

# Router for Data Parallelism 用于数据并行的路由器#
给定多个 GPU 运行多个 SGLang 运行时，SGLang Router 通过其独特的缓存感知负载均衡算法将请求分配到不同的运行时。该路由器是一个独立的 Python 包，可以用作 OpenAI API 的直接替代品
https://docs.sglang.ai/router/router.html

# Multi-Node Deployment 多节点部署# 
Run 405B (fp16) on Two Nodes
在两个节点上运行 405B （fp16）

```
# replace 172.16.4.52:20000 with your own node ip address and port of the first node

python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct --tp 16 --dist-init-addr 172.16.4.52:20000 --nnodes 2 --node-rank 0

python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct --tp 16 --dist-init-addr 172.16.4.52:20000 --nnodes 2 --node-rank 1
```
Note that LLama 405B (fp8) can also be launched on a single node.
请注意，LLama 405B （fp8） 也可以在单个节点上启动。
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 --tp 8
```
## Multi-Node Inference on SLURM
SLURM 上的多节点推理
例展示了如何通过 SLURM 跨多个节点为 SGLang 服务器提供服务。将以下作业提交到 SLURM 集群。

```
#!/bin/bash -l

#SBATCH -o SLURM_Logs/%x_%j_master.out
#SBATCH -e SLURM_Logs/%x_%j_master.err
#SBATCH -D ./
#SBATCH -J Llama-405B-Online-Inference-TP16-SGL

#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1  # Ensure 1 task per node
#SBATCH --cpus-per-task=18
#SBATCH --mem=224GB
#SBATCH --partition="lmsys.org"
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00

echo "[INFO] Activating environment on node $SLURM_PROCID"
if ! source ENV_FOLDER/bin/activate; then
    echo "[ERROR] Failed to activate environment" >&2
    exit 1
fi

# Define parameters
model=MODEL_PATH
tp_size=16

echo "[INFO] Running inference"
echo "[INFO] Model: $model"
echo "[INFO] TP Size: $tp_size"

# Set NCCL initialization address using the hostname of the head node
HEAD_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
NCCL_INIT_ADDR="${HEAD_NODE}:8000"
echo "[INFO] NCCL_INIT_ADDR: $NCCL_INIT_ADDR"

# Launch the model server on each node using SLURM
srun --ntasks=2 --nodes=2 --output="SLURM_Logs/%x_%j_node$SLURM_NODEID.out" \
    --error="SLURM_Logs/%x_%j_node$SLURM_NODEID.err" \
    python3 -m sglang.launch_server \
    --model-path "$model" \
    --grammar-backend "xgrammar" \
    --tp "$tp_size" \
    --dist-init-addr "$NCCL_INIT_ADDR" \
    --nnodes 2 \
    --node-rank "$SLURM_NODEID" &

# Wait for the NCCL server to be ready on port 30000
while ! nc -z "$HEAD_NODE" 30000; do
    sleep 1
    echo "[INFO] Waiting for $HEAD_NODE:30000 to accept connections"
done

echo "[INFO] $HEAD_NODE:30000 is ready to accept connections"

# Keep the script running until the SLURM job times out
wait
```
