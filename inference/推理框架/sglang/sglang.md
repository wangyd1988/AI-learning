- sglang 原理
- [x] https://zhuanlan.zhihu.com/p/28199298728
- sglang对deepseek的优化
- [x] https://docs.sglang.ai/references/deepseek.html
- sglang支持[request](https://docs.sglang.ai/backend/send_request.html)
- [x] 支持文本生成
- [x] 支持视觉语言模型(Vision Language Models)
- [x] 支持奖励模型(Reward Models)
- 访问
```
curl -s http://10.236.13.81:35519/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{{"model": "/var/lib/data/aicache/models/system/DeepSeek-R1-Distill-Qwen-7B", "messages": [{{"role": "user", "content": "What is the capital of France?"}}]}}'
```
- 启动参数
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
- 聊天参数
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
- 将模型转为特定格式
```
python convert.py --hf-ckpt-path /path/to/DeepSeek-V3 --save-path /path/to/DeepSeek-V3-Demo --n-experts 256 --model-parallel 16
```
- 参数
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


- SGLang 原生 API
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


