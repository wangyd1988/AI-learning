- SGLang
- [x] cpu-offload-gb
```
--cpu-offload-gb :How many GBs of RAM to reserve for CPU offloading.
sglang中meta设备

先在 meta 设备上创建模型，然后再将其移动到真实设备（如 cpu 或 cuda）并加载实际的权重。torch.nn.Module.to_empty() 方法可以将一个 meta 模型转换为空的、在目标设备上未初始化的模型，等待您填充权重。
```
- [x] [性能调优指南](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)

- VLLM [offload_weights](https://zhuanlan.zhihu.com/p/18942501855)
```
**offload_weights** 和 offload_activations：启用这些参数可以将模型权重和激活值部分卸载到 CPU，从而减少 GPU 显存占用。
```
- [x] [ How to offload some layers to CPU？](https://github.com/vllm-project/vllm/issues/3931) 
- [x] [Efficient LLM Inference with Activation Checkpointing and Hybrid Caching](https://arxiv.org/html/2501.01792v1)
- [x] (SAVING GPU MEMORY CRISIS WITH CPU OFFLOADING FOR ONLINE LLM INFERENCE)[https://arxiv.org/pdf/2411.01142]
- [x] [cpu-offload-gb](https://docs.vllm.ai/en/latest/cli/index.html?h=cpu+offload#cacheconfig)
```
GPU要卸载到CPU的大小，它默认值是0。这个可以看作是增加GPU内存大小的虚拟方式。例如，如果您有一个24 GB的GPU，当它设置为10时，
对外可以可以看做有34GB的GPU。这样就可以加载至少26GB的GPU内存的BF16的13B模型，需要注意，这需要快速的CPU-GPU互连，
因为模型的一部分在每个模型向前传递时从CPU内存加载到GPU内存。使用cpu offload时，需要uva (pin memory)，UVA（统一虚拟地址）和
固定内存（pin memory）是与内存管理相关的概念。在深度学习和计算机图形学中，固定内存可以提高数据传输的效率，因为它锁定在物理内存中，
防止被操作系统调页。
是将数据从GPU复制到CPU中
```

- pytorch
- [x] meta设备
```https://pytorch.ac.cn/docs/stable/meta.html
“meta” 设备是一个抽象设备，它表示一个只记录元数据而不包含实际数据的张量。Meta 张量有两个主要用例

模型可以加载到 meta 设备上，这允许您加载模型的表示而无需将实际参数加载到内存中。如果您需要在加载实际数据之前对模型进行转换，这将很有帮助。

大多数操作可以在 meta 张量上执行，生成新的 meta 张量，这些张量描述了如果您在真实张量上执行该操作会得到的结果。您可以使用它来执行抽象分析，而无需花费计算时间或空间来表示实际张量。由于 meta 张量没有真实数据，您无法执行依赖于数据的操作，例如 torch.nonzero() 或 item()。在某些情况下，并非所有设备类型（例如 CPU 和 CUDA）对于某个操作都具有完全相同的输出元数据；在这种情况下，我们通常倾向于忠实地表示 CUDA 的行为。
```
