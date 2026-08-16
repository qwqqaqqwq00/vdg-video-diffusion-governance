# 视频生成 DiT 跨设备数值鲁棒性研究报告
### 训练设备(NVIDIA bf16/fp8) ≠ 推理设备(MPS/消费NV/NPU) 的数值发散与修复/转换方案

> 调研日期：2026-08-13
> 方法论说明：本环境的 `web_search`（Brave）API 未配置，故全部检索改用 `web_fetch` 直连 **arXiv API**、**GitHub Issue 搜索页**、**项目 README** 完成——均为返回真实数据的实时网络调用，未凭记忆编造。凡引用 arXiv 论文均给出真实 arXiv ID 与发表日期；凡引用 issue 均给出真实编号与状态。少量"机理阈值"（如 bf16 7 位尾数、fp16 上限 65504）属 IEEE 754 事实，非编造。用户已定位的 MPS 三处 cast fp32 修复作为既定结论被引用并扩展。

---

## 0. 核心结论速览

1. 你遇到的"MPS 全黑视频/NaN"是 **跨设备数值发散**的典型实例，且 2026 年仍在复现：ComfyUI 官方 issue [#15315](https://github.com/Comfy-Org/ComfyUI/issues/15315)（2026-08-05 报告，08-10 关闭）记录 MiniMax H3 T2V 在 **M4 Max** 上"采样完成但输出全黑视频 + 音频 NaN"。同一机器 I2V/Ref2V 正常，**仅 T2V 失败**——这与你的发现一致：发散出现在特定条件路径（T2V 的文本条件调制链路）。
2. 发散根因可归为 6 类 op 机制（§1），其中 **softmax、AdaLN 调制、GELU-tanh、RMSNorm/LayerNorm、VAE 解码、时间步嵌入** 是高危点。MPS 侧有大量**已确认的内核缺陷**（§2），并非模型本身问题。
3. 你的"MPS 三处 cast fp32"修复可提炼为一条通用原则——**调制/归一化/非线性激活三类 op 在低精度后端强制 fp32 中间计算**——并已得到 2026 年量化论文的间接验证：Ideogram 4.0 INT8 量化（[2606.12280](https://arxiv.org/abs/2606.12280)）明确采用"**bf16 保护一小撮高脆弱层集**"，Wan2.1-T2V-14B 的 HiFloat8 量化（[2606.00957](https://arxiv.org/abs/2606.00957)）保留**前 2 + 后 3 个 boundary block 为 BF16**，与你的保 fp32 思路同源。
4. 训练侧桥接结论（§9）：要让模型天然适配异构端侧低精度，需在训练期引入 **量化感知 + AdaLN 调制参数远离精度悬崖的正则 + 关键 op 模拟低精度**。

---

## 1. 精度发散根因分类

训练 bf16/fp8(H100) → 推理 fp16/bf16/int8/int4(MPS/消费NV/NPU) 时，下列 op 易发散。每类给出机理与典型阈值。

### 1.1 Softmax / Attention

- **机理**：标准数值稳定 softmax = `exp(x - max(x)) / sum(exp(x - max(x)))`。问题出在 `x - max(x)` 这一步本身：当序列长、张量大时，**MPS 的 fused softmax 内核在该减法步直接产生 NaN**，即使所有元素相等（0.0598-0.0598=0 应得 0）。这是内核缺陷而非溢出。
- **证据**：PyTorch issue [#96602](https://github.com/pytorch/pytorch/issues/96602)（2023-03 开启，**至今 Open**，triaged，`module: mps`）。报告者 Birch-san 用分解 softmax 逐 op 断言，定位到 `diffs = x - maxes` 即引入 NaN；fp16 与 fp32 均失效；张量形状 `[10, 12416, 12416]`（大序列 attention）。
- **阈值/条件**：大序列长度（万级 token / 视频帧数 ×空间 token）触发；bf16/fp16 在 attention score 量级大时 `exp` 上溢或减法精度损失。fp16 下 `exp(x-max)` 当 score 差 > ~11 时下溢为 0（信息丢失，非 NaN）；MPS 则是内核 NaN。
- **fp8 训练侧**：FP4/FP8 attention 是当前 QAT 难点，[Attn-QAT](https://arxiv.org/abs/2603.00040)（2026-08 更新）指出"naive drop-in QAT（FP4 前向 + 高精度 FlashAttention 反向）导致训练不稳定"，需匹配反向低精度重算 + 解决 FA 梯度的隐式精度假设。

### 1.2 GELU / SiLU / GELU-tanh

- **机理（你已定位）**：MPS 上 bf16 融合 GELU(tanh 近似) 在 |x|≥15 即 NaN——**非溢出，是 Metal 融合内核缺陷**；fp16 则 `x³` 在 |x|>40 溢出 65504→Inf→NaN。CPU bf16 正常，证明 MPS 特有。SiLU(`x*sigmoid(x)`) 相对稳健但大负 x 时 sigmoid 饱和、bf16 尾数仍可能丢精度。
- **阈值**：GELU-tanh 近似含 `tanh(√(2/π)(x + 0.044715x³))`；fp16 的 `x³` 上限受 65504 约束 ⇒ |x| ≲ 40.3；bf16 动态范围大（≈±3.4e38）但 7 位尾数使中间多项式精度差，MPS 融合内核在 |x|≥15 出 NaN。
- **修复**：MPS 强制 GELU cast fp32 计算（你的方案）。通用化见 §7。

### 1.3 AdaLN 调制（1+scale 灾难性抵消）★最关键

- **机理（你已定位）**：DiT 的 AdaLN 做 `rms_norm(x)*(1+scale)+shift`。当 `scale ≈ -1` 时，`(1+scale) → 0`，**bf16 的 7 位尾数使 (1+scale) 丢失全部有效数字**（catastrophic cancellation），后续乘法把信号清零，gate_msa/gate_mlp 若也接近 0 则整层输出趋零 → 累积成黑帧。
- **阈值/条件**：`|1+scale| < 2^-7 ≈ 0.0078` 时 bf16 已无法表示相对误差；`scale ∈ [-1.0, -0.992]` 即进入灾难区。fp16（10 位尾数）阈值更宽（`|1+scale| < 2^-10 ≈ 9.8e-4`），但仍可能在 scale 极接近 -1 时丢精度。
- **为何 T2V 比 I2V 更易触发（对应 #15315 现象）**：T2V 的文本条件经 timestep + text embedding 双重调制，调制参数动态范围更大、更易出现 scale≈-1 的极端步；I2V/Ref2V 的图像条件路径调制分布更温和。这解释了 #15315 中"仅 T2V 全黑"。
- **修复**：AdaLN 自注意力调制（scale_msa/shift_msa/gate_msa）与 MLP 调制（scale_mlp/shift_mlp/gate_mlp）三处强制 fp32（你的方案）。训练侧正则见 §4.3、§9。

### 1.4 RMSNorm / LayerNorm

- **机理**：RMSNorm `x / sqrt(mean(x²)+eps)`；方差计算涉及平方和，大激活下 `x²` 易在 fp16 溢出（>65504 ⇒ |x|>256），bf16 范围大但尾数少导致 `mean(x²)` 精度差、除法放大误差。LayerNorm 还多一步减均值。
- **证据**：PyTorch issue [#96113](https://github.com/pytorch/pytorch/issues/96113)（2023-03 关闭，2.0.1 修复）："[mps] [PyTorch 2.0] LayerNorm crashes when input is in float16"——MPS LayerNorm fp16 直接崩溃。
- **阈值**：fp16 下 `|x| > sqrt(65504) ≈ 256` 即平方溢出；bf16 下 eps 若过小（<1e-5）与方差量级失配会放大误差。
- **修复**：norm 内部（方差、除法）fp32 计算，仅输入输出 cast。

### 1.5 VAE 解码

- **机理**：VAE decoder 数值范围大（latent → 像素 [-1,1]），含 GroupNorm（小 group 下统计量不稳定）、上采样卷积、SiLU；量化后激活范围漂移使 GroupNorm 的 mean/var 失准，输出色彩偏移/块状伪影/黑帧。VAE 通常**不与 DiT 同步量化**——社区共识是 VAE 保高精度。
- **证据**：diffusers issue [#6336](https://github.com/huggingface/diffusers/issues/6336)（TensorRT pipeline 编译 VAE 报错）；ComfyUI-GGUF README 明确 TE/VAE 与 DiT 分离处理（§3.1）。
- **阈值**：VAE latent 量级 ~O(1)–O(10)，像素输出 [-1,1]，中间特征可达 O(100)，fp16 在 O(100) 附近平方溢出。
- **修复**：VAE 解码全程 fp16/fp32（不 int8）；GroupNorm/上采样关键步 fp32。

### 1.6 时间步嵌入（timestep embedding）

- **机理**：sinusoidal timestep embedding 经 MLP 投影到调制参数； timestep 作为标量经频率展开后量级跨度大，低精度下投影 MLP 易放大误差并直接喂给 AdaLN（与 §1.3 耦合——timestep 投影误差正是导致 scale≈-1 的来源之一）。
- **证据**：diffusers issue [#11456](https://github.com/huggingface/diffusers/issues/11456)（Open，2025-04）："onnx export failure - timestep parameter with static value"——timestep 参数在静态图导出/低精度下行为异常。
- **修复**：timestep embedding MLP 保 fp32；导出时 timestep 用动态 shape 而非静态常量。

### 发散点 → 机理 → 阈值/条件 → 修复 → 适用设备（汇总见 §8 表）

---

## 2. MPS / Apple Silicon 数值坑（扩展你的发现）

### 2.1 已确认的 MPS 数值/内核缺陷（PyTorch issue 追踪）

下列均为 `module: mps` / `module: NaNs and Infs` 标签的真实 issue，证明 MPS 数值缺陷是**系统性、持续到 2026 年**的问题，而非个例：

| Issue | 状态 | 缺陷 |
|---|---|---|
| [#96602](https://github.com/pytorch/pytorch/issues/96602) | Open (2023) | softmax 大张量 NaN，减法步引入 |
| [#96113](https://github.com/pytorch/pytorch/issues/96113) | Closed 2.0.1 | LayerNorm fp16 崩溃 |
| [#192577](https://github.com/pytorch/pytorch/issues/192577) | Closed 2026-08-10 | multinomial 返回零概率索引；`exponential_()` 产生 -0.0，Gumbel fast path 变 NaN |
| [#192507](https://github.com/pytorch/pytorch/issues/192507) | Closed 2026-08-08 | `torch.hypot` 对 inf 输入溢出返回 NaN（CPU 不这样） |
| [#190476](https://github.com/pytorch/pytorch/issues/190476) | Closed 2026-07-20 | `torch.nextafter` 对 bfloat16 是 no-op |
| [#190057](https://github.com/pytorch/pytorch/issues/190057) | Closed 2026-07-19 | 多层 LSTM dropout=1.0 返回全 NaN |
| [#189243](https://github.com/pytorch/pytorch/issues/189243) | Closed 2026-07-08 | `torch.div(rounding_mode="floor")` 非有限数语义与 CPU/CUDA 不同 |
| [#188756](https://github.com/pytorch/pytorch/issues/188756) | Closed 2026-08-03 | `index_select`/`F.embedding` 源张量 >2³¹ 元素时静默返回全零行 |
| [#187521](https://github.com/pytorch/pytorch/issues/187521) | Closed 2026-06-17 | `baddbmm` 当 beta==0 时错误传播 NaN |
| [#190632](https://github.com/pytorch/pytorch/issues/190632) | Closed 2026-07-21 | **训练时 backward() 产生非确定性 NaN/Inf 梯度**（Faster R-CNN） |
| [#187455](https://github.com/pytorch/pytorch/issues/187455) | Open (2026-06) | 将剩余 MPSGraph op 迁移到原生 Metal 以约束 graph-cache 内存 |

**关键启示**：`#188756`（embedding 全零）与 `#187521`（baddbmm beta==0 NaN）与你的 AdaLN 灾难性抵消同构——都是"特定数值条件触发静默错误输出"。`#190632` 证明 MPS 在**训练**反向也非确定性，意味着 MPS 不宜作为数值回归基线。

### 2.2 "invalid value encountered in cast" 与 bf16 尾数

- 该 warning 由 `bf16→fp16` 或含 Inf/NaN 张量的 cast 触发。搜索 PyTorch issue（`MPS "invalid value encountered in cast"`）命中 [#96602](https://github.com/pytorch/pytorch/issues/96602) 与 [#96113](https://github.com/pytorch/pytorch/issues/96113)——即 cast warning 常是**上游 op 已产生 NaN/Inf 的下游表现**，而非 cast 本身出错。排查时应向上回溯到 §1 的某个 op。
- bf16（7 位尾数）→ fp16（10 位尾数）cast 时，bf16 能表示的值 fp16 多数能表示，但**bf16 计算中间产物若已因尾数不足丢精度，cast 后无法恢复**。反之 fp16→bf16 会进一步截断尾数。

### 2.3 Metal fusion 内核缺陷

- 你的发现（bf16 融合 GELU-tanh |x|≥15 NaN）属于此类。MPS 的 MPSGraph 会把 `tanh(√(2/π)(x+0.044715x³))` 融合为单一 Metal 内核，融合中间不提升精度，`x³` 项在 bf16 下精度不足且内核未做范围保护。
- [#187455](https://github.com/pytorch/pytorch/issues/187455) 表明 PyTorch 正在把 MPSGraph 融合 op 迁移到原生 Metal，暗示**融合内核的数值正确性仍是进行中的工程问题**。
- ComfyUI-GGUF README 记录：MacOS Sequoia 上 **torch 2.6.X nightly 触发 "M1 buffer is not large enough" 错误，需退回 torch 2.4.1**——MPS 后端版本敏感。

### 2.4 MLX 是否比 MPS backend 更稳？

**结论：MLX 通常更稳且更适合 Apple Silicon 部署，但生态覆盖仍不及 PyTorch。** 证据：

- [Production-Grade Local LLM Inference on Apple Silicon](https://arxiv.org/abs/2511.05502)（2025-10）系统对比 MLX / MLC-LLM / llama.cpp / Ollama / PyTorch MPS，结论：**"PyTorch MPS 在大模型与长上下文上仍受内存约束限制"**，**MLX 取得最高持续生成吞吐**。MLX 由 Apple 官方维护，针对统一内存与 Metal 重新实现算子（非走 MPS 的 MPSGraph 融合路径），避开了大量 MPS 历史包袱。
- [Benchmarking On-Device ML on Apple Silicon with MLX](https://arxiv.org/abs/2510.18921)（2025-10）：MLX transformer 推理延迟对比 PyTorch，MLX 在 Apple 生态内更优。
- [Systematic Optimization of Real-Time Diffusion Model Inference on Apple M3 Ultra](https://arxiv.org/abs/2605.16259)（2026-02）：10 阶段优化实验，揭示 **"CUDA 上成立的优化在 Apple Silicon 统一内存上未必有效——量化无加速、并行推理无效、神经引擎不适合大模型"**，最终用 **CoreML 转换 SDXS-512** 达 22.7 FPS。
- **建议**：视频 DiT 数值敏感场景，优先评估 MLX 后端（如 mlx-vlm / 社区 mlx 视频项目）作为 MPS 的替代；MLX 的算子实现更可控、可逐 op 指定精度。但需注意 MLX 对自定义 attention/视频专用算子覆盖度可能不足。

### 2.5 社区 workaround 汇总

1. **关键 op cast fp32**（你的方案；Hunyuan3D/HiDream 亦有同类 workaround——你已引用，此处不重复编造细节）。
2. **退回 torch 2.4.1**（ComfyUI-GGUF README，规避 2.6.X buffer bug）。
3. **CPU fallback 特定 op**：对 MPS 反复出问题的 op（如 #96602 softmax），临时 `.to('cpu')` 计算再迁回，代价是性能。
4. **禁用 MPS 融合 / 用 eager 分解 op**：把融合 GELU 拆成 `cast fp32 → tanh 近似 → cast 回`，正是你的做法。
5. **切 MLX 后端**（§2.4）。
6. **黑帧/NaN 自动检测 + 重试**（§5.2）。

---

## 3. 模型转换管线（训练 → 端侧）

### 3.1 safetensors(bf16/fp32) → GGUF（Q4_K_M / Q5 / Q8_0）用于 DiT

- **工具**：[city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)（3.9k star），`tools/` 目录提供量化脚本；依赖 `gguf` 库。llama.cpp 的 GGUF 量化思路（k-quants、imatrix）迁移到扩散模型。
- **关键结论（README 原文）**："量化对常规 UNET（conv2d）不可行，但 **transformer/DiT 模型（如 flux）受量化影响较小**，可在更低 bit-per-weight 上运行。"——DiT 的线性层主导、卷积少，故 GGUF 量化友好。
- **量化哪些层**：DiT 的所有 `nn.Linear`（QKV、FFN、proj）；**TE（文本编码器，如 T5）单独量化**（README 明确支持 `t5_v1.1-xxl GGUF`，用 `*CLIPLoader (gguf)` 节点）；**VAE 单独保留高精度**（不进 GGUF）。
- **常见坑**：
  - MacOS Sequoia 需 torch 2.4.1（2.6.X "M1 buffer is not large enough"）。
  - Q4_K_M 是质量/体积平衡点；Q8_0 接近无损。低比特（Q4 以下）在 AdaLN 调制路径易触发 §1.3。
  - LoRA 加载实验性支持。
- **2026 验证**：[Holding the FP8 Quality Ceiling](https://arxiv.org/abs/2606.12280)（Ideogram 4.0，2026-06）显示 **GGUF Q4_K 在 standalone 质量上与 NF4 基线持平，是质量-内存 Pareto 前沿选择**；[HyperQuant](https://arxiv.org/abs/2606.23406)（2026-06）量化 **19B LTX-2 视频 DiT**"无可观察的逐帧伪影"（4 bps，Hadamard+格点+Rice 编码）。

### 3.2 → CoreML / MLPackage

- **工具**：`coremltools`（[apple/coremltools](https://github.com/apple/coremltools)，5.4k star）的 `ct.convert()`，`UnifiedConversionProtocol`；`FlexShape`/`RangeDim` 做 flexible shape；`compute_precision=ct.precision.FLOAT16` 或 `ct.precision.FLOAT32`。
- **ANE 适配与算子覆盖坑**：
  - [#1763](https://github.com/apple/coremltools/issues/1763)（closed 2023-05）：**"Apple's ANE Optimised MultiHeadAttention Export Fails With Flexible Input Shape"**——ANE 优化的 MHA 在 flexible shape 下导出失败，这是 DiT 转换 CoreML 的核心障碍（attention 是 DiT 主干）。
  - [#1359](https://github.com/apple/coremltools/issues/1359)：**Einsum 不支持的方程格式**——很多自定义 attention 用 Einsum 表达，CoreML 不覆盖。
  - [#1947](https://github.com/apple/coremltools/issues/1947)：`MLMultiArray of data type Float16 is not supported`（旧版）。
  - ANE 偏好 fp16/int8，但大模型常被静默调度到 CPU/GPU（[ANEForge](https://arxiv.org/abs/2606.17090) 2026-06 指出 "CoreML 把 ANE 当调度选项，模型可能静默跑在 CPU/GPU"）。
- **M3 Ultra 实证**：[2605.16259](https://arxiv.org/abs/2605.16259) 用 CoreML 转 SDXS-512 达实时，但结论是"**神经引擎不适合大规模模型、量化无加速**"——CoreML 路线对大视频 DiT 仍受限。
- **ANE 直连新方向**：[ANEForge](https://arxiv.org/abs/2606.17090)（2026-06）绕过 CoreML 直接编程 ANE，支持 int8/int4/稀疏权重与原生 fused attention，已验证 Stable Diffusion U-Net 前向——未来或可用于 DiT ANE 部署。
- **建议**：DiT → CoreML 需把 attention 改写为 CoreML 原生 op（避免 Einsum/自定义）、flexible shape 限定固定档位、调制/norm 层标 `fp32` precision。

### 3.3 → ONNX

- **难点**：自定义 attention（FlashAttention/SDPA 不可直接导出）、动态 shape（视频帧数/分辨率可变）、timestep 动态参数、控制流。
- **证据**：
  - diffusers [#11456](https://github.com/huggingface/diffusers/issues/11456)（Open）：**ONNX 导出失败——timestep parameter with static value**——timestep 被固化为静态常量，破坏多步采样。
  - diffusers [#7922](https://github.com/huggingface/diffusers/issues/7922)：`ORTStableDiffusionXLPipeline`（ONNX Runtime SDXL 路径存在但脆弱）。
  - [2603.01023](https://arxiv.org/abs/2603.01023)（2026-03）：用 **ONNX GraphSurgeon 把 18,398 节点的单体扩散 planner 拆成 3 个独立模块**，并在原生 C++ 重实现 DPM-Solver++ 去噪循环——这是大 DiT 导出 ONNX 的可行范式：**模块化拆分 + 去噪循环外置**。
- **关键步骤**：① attention 用 `ScaledDotProductAttention` ONNX op 或手写 QKV softmax（注意 §1.1 MPS 类似的 softmax 数值问题在 ONNX Runtime EP 上也需 fp32 softmax）；② 用 dynamic_axes 声明 batch/seq/timestep；③ 把 sampler/去噪循环留在 PyTorch 侧，仅导出单步 UNet/DiT。
- **常见坑**：FP16 ONNX 在 CPU EP 上 GroupNorm/softmax 精度差；需 `ort.SessionOptions.graph_optimization_level` 谨慎；TensorRT EP 对动态 shape 支持差。

### 3.4 → INT8 校准（TensorRT PTQ / AIMET）

- **TensorRT**：diffusers [#11032](https://github.com/huggingface/diffusers/issues/11032)（closed 2025-11）"Support TRT as a backend"；[#6336](https://github.com/huggingface/diffusers/issues/6336) TensorRT 编译 VAE 报错。TRT INT8 需校准数据集（代表性 prompt+timestep 组合），calibrator 跑前向收集激活 min/max。
- **消费 NV 的 INT8 软件陷阱**：[Realizing Native INT8 Compute](https://arxiv.org/abs/2606.14598)（Ideogram 4.0，2026-06）揭露——**生产级 "INT8" 前向其实只是把权重/激活量化后立即反量化回 bf16 跑 bf16 matmul，从未真正调用 INT8 tensor core**，故消费 Ampere 上 INT8 反而比 FP8/NF4 慢。修复用 fused Triton INT8 GEMM（int8×int8→int32，per-token×per-channel 反量化，bias 折入 epilogue），余弦相似度 1.0 无 NaN，2.8–4.2x 单 GEMM 加速。**注意：A100/B200 上同内核输给原生 bf16/FP8 路径**——消费卡与数据中心卡的量化收益截然不同。
- **INT8 + bf16 保护高脆弱层**：[2606.12280](https://arxiv.org/abs/2606.12280) Ideogram 4.0 INT8 W8A8 配方 = per-channel 权重 + per-token 动态激活 + SmoothQuant + **bf16 保护一小撮高脆弱层集**，在 CLIP/PickScore 上与 FP8 统计无差异，LPIPS 0.243（最忠实复现 FP8 输出）。用 **PSNR/SSIM/LPIPS** 对比 FP8 reference 做保真度验证。
- **AIMET**（Qualcomm）：PTQ/SQAT/AdaRound，适合 NPU（Hexagon）int8/int4；校准数据集需覆盖 timestep 分布（DiT 激活随 timestep 漂移）。
- **常见坑**：校准集未覆盖高噪声 timestep → 该步激活 outlier 未被吸收 → 黑帧；per-tensor 量化在 outlier 通道下崩坏，需 per-channel 或 SmoothQuant。

---

## 4. 数值鲁棒技术

### 4.1 关键 op 保 fp32（敏感性经验）

基于 §1 与 2026 量化论文的共识，**必须高精度**的层/op：

| op/层 | 为何敏感 | 保精度方式 |
|---|---|---|
| GELU-tanh / GELU | 多项式 + tanh，bf16 融合内核缺陷 | fp32 计算后 cast 回 |
| AdaLN 调制 (1+scale) | 灾难性抵消 | 调制参数与乘加 fp32 |
| RMSNorm / LayerNorm | 平方和/方差/除法 | 统计与除法 fp32 |
| Softmax | 大序列减法 NaN/下溢 | fp32 softmax（或 flash-attn fp32 路径） |
| VAE 解码（GroupNorm/上采样） | 数值范围大 | 全程 fp16/fp32，不 int8 |
| 时间步嵌入 MLP | 喂给 AdaLN，误差放大 | MLP fp32 |
| Attention score（QK^T） | 大 score 溢出 | fp32 或缩放后 fp16 |

**层级别**：boundary block（首尾各几个 transformer block）最脆弱。[2606.00957](https://arxiv.org/abs/2606.00957) Wan2.1-T2V-14B 保留前 2 + 后 3 block 为 BF16；[2606.12280](https://arxiv.org/abs/2606.12280) 用 bf16 保护"高脆弱层集"。**这与你的"三处 cast fp32"是同一原则的两种粒度**：op 级 vs block 级。

### 4.2 损失量化 vs 无损量化 / per-channel vs per-tensor / SmoothQuant / AWQ / SVDQuant

- **per-tensor**：一个 scale 覆盖全层，outlier 通道拖垮整体 → DiT 上灾难性。**per-channel（权重）+ per-token（激活）** 是 W8A8 基本要求（[2606.14598](https://arxiv.org/abs/2606.14598)、[2606.12280](https://arxiv.org/abs/2606.12280)）。
- **SmoothQuant**（[2211.10438](https://arxiv.org/abs/2211.10438)，ICML 2023）：把激活 outlier 的量化难度**等价迁移到权重**（`x/s · s·W`），让两侧都好量化。W8A8 主力方法。缺点：改变通道量级可能损害精度（[KroQuant 2607.21446](https://arxiv.org/abs/2607.21446) 指出）。
- **AWQ**（activation-aware weight quantization，源自 LLM）：权重-only 量化，用激活幅度识别显著通道并加 scale 保护。2026 LLM 仍广泛使用（[AAAC 2605.08692](https://arxiv.org/abs/2605.08692) 以 AWQ/GPTQ 为基线）。对 DiT，**权重-only 不足以解决激活 outlier**，需配合 SmoothQuant/SVDQuant。
- **SVDQuant**（[2411.05007](https://arxiv.org/abs/2411.05007)，ICLR 2025 Spotlight）：4-bit 范式，**用低秩分支吸收 outlier**（先平滑迁移激活 outlier 到权重，再 SVD 取权重 outlier 到高精度低秩分支，残差走低 bit）。Nunchaku 推理引擎融合低秩与低 bit 内核。FLUX.1 12B 内存降 3.5x，RTX 5090 用 NVFP4 达 3.1x。**Timestep-Aware SVDQuant-GPTQ**（[2605.27003](https://arxiv.org/abs/2605.27003)，2026-05）扩展到 Wan2.2-I2V MoE DiT，按 expert × timestep-bin 独立校准。
- **Hadamard 旋转类**（2026 主流）：[OrbitQuant 2607.02461](https://arxiv.org/abs/2607.02461)（data-agnostic，旋转基下固定边际，FLUX/Wan2.1/CogVideoX，推到 W2A4）、[HyperQuant 2606.23406](https://arxiv.org/abs/2606.23406)（LTX-2 19B，4 bps 近无损）、[KroQuant 2607.21446](https://arxiv.org/abs/2607.21446)（Kronecker 块变换）、[DiRotQ 2605.16732](https://arxiv.org/abs/2605.16732)（PCA 子空间高精度）。旋转把未知分布拉成近高斯，缓解 outlier——对发散有根本缓解。
- **混合精度/时间步感知**：[AdaTSQ 2602.09883](https://arxiv.org/abs/2602.09883)（Pareto 时间步动态位宽）、[6Bit-Diffusion 2603.18742](https://arxiv.org/abs/2603.18742)（NVFP4/INT8 时间步混合）、[SemanticDialect 2603.02883](https://arxiv.org/abs/2603.02883)（block 混合格式）。**核心洞见：DiT 量化敏感度随 timestep 强烈变化**，静态位宽次优。
- **无损量化**：[HyperQuant](https://arxiv.org/abs/2606.23406) 在 4 bps 声称"近无损"（bit-stripping + 熵编码 Rice）；[TreeQ 2512.06353](https://arxiv.org/abs/2512.06353) "首个 DiT 上近无损 4-bit PTQ"。真正无损通常需 ≥8 bit 或高精度低秩补偿。

### 4.3 训练侧选择提升下游鲁棒

- **QAT（量化感知训练）**：
  - [EfficientDM 2310.03270](https://arxiv.org/abs/2310.03270)（ICLR 2024）：QALoRA（可并入权重的量化低秩适配器）+ data-free 蒸馏 + scale-aware 优化 + 时间步学习步长量化，W4A4 仅 +0.05 sFID。
  - [QVGen 2505.11497](https://arxiv.org/abs/2505.11497)（ICLR 2026）：**视频** DM 的 QAT，证明**降低梯度范数是收敛关键**，用辅助模块 Φ 缓解大量化误差 + rank-decay 消除推理开销；CogVideoX-2B 3-bit 在 VBench Dynamic Degree +25.28。
  - [HQ-DM 2512.05746](https://arxiv.org/abs/2512.05746)：单 Hadamard 变换 QAT，减激活 outlier 且不放大权重 outlier，支持 INT 卷积。
  - [RobuQ 2509.23582](https://arxiv.org/abs/2509.23582)（ICML 2026）：W1.58A2，Hadamard 把 per-token 分布转正态。
  - [Attn-QAT 2603.00040](https://arxiv.org/abs/2603.00040)：FP4 attention QAT 两原则（反向低精度重算匹配 + 解决 FA 梯度隐式精度假设）。
- **蒸馏到量化友好学生**：[EfficientDM](https://arxiv.org/abs/2310.03270) 即蒸馏范式；[2605.16259](https://arxiv.org/abs/2605.16259) 用蒸馏模型 SDXS-512 配 CoreML 达实时。把大 DiT 蒸馏到小而量化友好的学生，比直接量化大模型更稳。
- **AdaLN scale 范围正则（避免 scale≈-1）★你的洞见**：在训练损失加正则项，约束 `scale` 远离 -1（如 `λ · mean(relu(|1+scale| - τ))` 或对 scale 加 tanh 中心化偏置使分布远离 -1）。机理上直接消除 §1.3 的灾难性抵消源。**未见专门论文做此正则**（这是可发表的研究点），但与 [2606.00957](https://arxiv.org/abs/2606.00957) "boundary block 保高精度"互补：前者训练期消除危险区，后者推理期保护危险层。
- **低精度训练使学生量化友好**：用 bf16/fp8 训练（[FPSAttention 2506.04648](https://arxiv.org/abs/2506.04648) 在 Wan2.1 上做 training-aware FP8+稀疏）使模型权重天然适应低精度激活分布。注意 [AIS 2605.13907](https://arxiv.org/abs/2605.13907) 警告：**FP8 rollout 与 BF16 trainer 的不匹配会偏置梯度、甚至训练崩溃**——低精度训练需重要性采样校正。

### 4.4 随机舍入 / 模拟低精度训练

- **随机舍入（stochastic rounding）**：量化时以 `frac(x/s)` 的小数部分为概率向上舍入，期望无偏。在 bf16/fp8 训练累积与低 bit 量化中减小系统性偏差。属成熟技术（GPU 上实现开销较大，NPU/ANE 支持有限）。
- **模拟低精度训练（fake quant / QAT emulate）**：前向插 `quant_dequant` 模拟目标精度，反向用 STE（直通估计器）。即 §4.3 QAT 的底层机制。建议**用端侧真实精度格式模拟**（如目标 MPS 走 bf16+fp32 关键 op，则训练期就用同样混合精度模拟），使训练分布与推理一致——这正是 [2412.06661](https://arxiv.org/abs/2412.06661) "Serial-to-Parallel pipeline 维持训练-推理一致性"的要点。

---

## 5. 跨设备验证方法

### 5.1 输出对比（SSIM / PSNR / LPIPS）

- [2606.12280](https://arxiv.org/abs/2606.12280) 用 **PSNR/SSIM/LPIPS** 对比量化输出与 FP8 reference（最高精度公开 checkpoint），发现"standalone 质量分数齐平时，**保真度与文字渲染**才是分离指标"——即不能只看 CLIP/PickScore，必须看 LPIPS/SSIM 这类逐像素/感知保真度。
- [DiRotQ 2605.16732](https://arxiv.org/abs/2605.16732) 用 FID + PSNR（MJHQ-30K）。
- [HyperQuant 2606.23406](https://arxiv.org/abs/2606.23406) 用"无可观察逐帧伪影"（LTX-2 19B）。
- **建议协议**：固定 prompt/timestep/seed 网格，CPU fp32 作 ground truth，目标设备输出算 PSNR/SSIM/LPIPS；设阈值（如 LPIPS < 0.3、SSIM > 0.9）作回归门禁。

### 5.2 黑帧 / NaN 自动检测

- **NaN/Inf 检测**：每步采样后 `torch.isnan/ isinf(x).any()` 断言；ComfyUI [#15315](https://github.com/Comfy-Org/ComfyUI/issues/15315) 的报错 `avcodec_send_frame(): [aac] Input contains (near) NaN/+-Inf` 即编码器侧自动捕获——说明**音视频编码器是天然的 NaN 后置探测器**。
- **黑帧检测**：输出帧 `mean/std`，若全帧均值≈0 且 std≈0 即黑帧；或 SSIM(vs 随机噪声) 异常高（说明输出退化为常数）。
- **回归测试集**：维护一组"已知易触发"prompt（高动态、长序列、强文本条件）+ 固定 seed，每次转换/更新后跑全量，对比基线。

### 5.3 逐 op 数值探针（CPU vs 目标设备）★你的第 3 节方法

- **方法论**：对每个 op，同输入在 CPU fp32 与目标设备（MPS bf16/fp16/int8）分别计算，对比 `max_abs_diff / mean_abs` 与 `nan_count`。这正是 [PyTorch #96602](https://github.com/pytorch/pytorch/issues/96602) 报告者用的"分解 softmax 逐 op 断言"技术——**社区已验证的定位范式**。
- **工程化**：hook 每个 `nn.Module.forward`，记录输入输出张量到 CPU，离线对比；或用 `torch.autograd.profiler` + 自定义 anomaly 检测。
- **建议**：构建"数值探针套件"覆盖 §1 六类 op，对每类注入边界输入（如 scale=-0.999、|x|=20 的 GELU、大序列 softmax、大激活 RMSNorm），自动判定目标设备是否发散。

### 5.4 确定性 / 可复现性

- PyTorch [#190632](https://github.com/pytorch/pytorch/issues/190632) 证明 **MPS backward 产生非确定性 NaN/Inf 梯度**——MPS 不宜作数值基线，**应以 CPU fp32 为确定性基线**。
- 启用 `torch.use_deterministic_algorithms(True)`（MPS 支持有限），固定 seed，关闭 cudnn benchmark。
- 回归测试需记录 torch/驱动/macOS 版本（MPS 版本敏感，见 §2.3）。

---

## 6. fp8 训练对端侧的影响

Hopper/Blackwell fp8 训练产出的权重，在消费级 Blackwell（fp8 推理）与 MPS/NPU（只能 int8/int4/bf16）上发散差异：

- **训练-推理格式错配**：训练用 fp8（E4M3 前向 / E5M2 反向，[Efficient FP8 PTQ 2309.14592](https://arxiv.org/abs/2309.14592) 指出 E4M3 适合 NLP、E3M4 略优 CV、FP8 覆盖 92.64% 工作负载 vs INT8 65.87%）。端侧 MPS 无 fp8 硬件 → 必转 bf16/int8。fp8 权重的动态范围假设（E4M3 max≈448）与 int8/bf16 不同，直接 cast 会丢范围信息。
- **消费 Blackwell vs 数据中心 Blackwell**：[2606.14598](https://arxiv.org/abs/2606.14598) 揭示 A100/B200 的原生 bf16/FP8 路径太快，使 INT8 fused kernel 反而落后；而消费 Ampere（RTX 3090）无 FP8 tensor core，INT8 fused kernel 才领先。**结论：同一量化方案在消费卡与数据中心卡上收益相反**，端侧策略需独立选型。
- **NVFP4 / MXFP4 新格式**：[SVDQuant 2411.05007](https://arxiv.org/abs/2411.05007) 在 RTX 5090（Blackwell）用 NVFP4 达 3.1x；[6Bit-Diffusion 2603.18742](https://arxiv.org/abs/2603.18742) 用 NVFP4/INT8 混合。但 MPS/NPU 无 NVFP4 → 这类训练成果**无法直接迁移到 Apple/NPU**，需退到 int8/int4 + §4.1 关键 op 保 fp32。
- **跨格式发散点**：[HyperQuant 2606.23406](https://arxiv.org/abs/2606.23406) 发现"**在 post-RHT 格点输出上 int8 击败 fp8**"——即 Hadamard 旋转后 int8 反而更准。暗示 fp8 训练权重转 int8 推理时，加旋转预处理可缓解发散。
- **fp8 训练不匹配风险**：[AIS 2605.13907](https://arxiv.org/abs/2605.13907) 警告 FP8 rollout 与 BF16 trainer 不匹配会偏置梯度、训练崩溃——若训练期未妥善处理，产出的权重本身带数值偏置，下游端侧更易发散。
- **缓解**：① 训练期用 QAT 模拟端侧目标精度（§4.3）；② 导出时按端侧能力分档（Blackwell→NVFP4/fp8；消费 Ampere→INT8 fused；MPS/NPU→int8/int4 + bf16 关键层）；③ 用 Hadamard 旋转预处理权重降低对格式的敏感度；④ 关键 op（§4.1）任何格式都保高精度。

---

## 7. 通用跨设备数值鲁棒补丁模板（提炼你的 MPS 三处 cast fp32）

你的修复（MPS 上对 **GELU、AdaLN-自注意力调制、AdaLN-MLP 调制**三处强制 cast fp32 计算后 cast 回）可提炼为以下**设备无关**的补丁模板，可复用于消费 NV / NPU / CoreML / ONNX：

### 模板原则
> **在低精度后端上，凡含"加性抵消风险（1+scale）、多项式非线性（GELU/SiLU）、归一化统计（Norm）、大序列归约（softmax）、大数值范围（VAE）"的 op，其敏感中间计算强制 fp32，仅边界张量按后端精度 cast。**

### 伪代码（PyTorch 风格，可适配 CoreML/ONNX）

```python
# 设备能力探测：决定哪些 op 需保 fp32
PREC_GUARD_OPS = {
    "gelu", "gelu_tanh", "silu",          # 多项式/非线性
    "adln_modulate",                        # (1+scale) 抵消
    "rmsnorm", "layernorm", "groupnorm",    # 归一化统计
    "softmax", "attention_scores",          # 大序列归约
    "timestep_embed_mlp",                   # 喂给 AdaLN
    "vae_decode",                            # 大数值范围
}
KEEP_FP32 = (device == "mps") or (backend in {"int8","int4","coreml_ane"})
# NPU/消费NV int8 同样建议保 fp32；Blackwell fp8 可放宽

def guarded_op(op, x, *args, **kw):
    if op in PREC_GUARD_OPS and KEEP_FP32:
        orig_dtype = x.dtype
        x32 = x.float()                         # cast 入 fp32
        y32 = op(x32, *args, **kw)               # fp32 计算
        return y32.to(orig_dtype)                # cast 回
    return op(x, *args, **kw)

# AdaLN 调制专项（核心）
def adln_modulate(x, scale, shift, gate=None):
    if KEEP_FP32:
        x32, s32, sh32 = x.float(), scale.float(), shift.float()
        # 关键：先算 (1+scale) 在 fp32，避免 bf16 抵消
        y = rms_norm_fp32(x32) * (1.0 + s32) + sh32
        if gate is not None:
            y = y * gate.float()
        return y.to(x.dtype)
    return rms_norm(x) * (1 + scale) + shift
```

### 各后端落地
- **MPS**：你的三处（GELU、scale_msa/shift_msa/gate_msa、scale_mlp/shift_mlp/gate_mlp）+ 扩展 RMSNorm/softmax/timestep/VAE。
- **消费 NV (int8)**：同模板；配合 SmoothQuant + per-channel/per-token + boundary block 保 bf16（[2606.12280](https://arxiv.org/abs/2606.12280)、[2606.00957](https://arxiv.org/abs/2606.00957)）。
- **NPU (Ascend/Hexagon)**：HiFloat8/int8 + 首尾 block 保 bf16（[2606.00957](https://arxiv.org/abs/2606.00957) Wan2.1-T2V-14B 在 Ascend 910B 的 boundary-protection 即此模板的 block 级实例）。
- **CoreML/ANE**：`compute_precision` 对上述 op 指定 FLOAT32；attention 改原生 op（规避 [#1763](https://github.com/apple/coremltools/issues/1763) flexible shape 失败）。
- **ONNX**：导出时对上述 op 强制 fp32 节点；softmax 用 fp32 ONNX op；timestep 动态 axes。

### 与量化研究的对应
此模板 = **op 级保精度**；[2606.00957](https://arxiv.org/abs/2606.00957) boundary-block 保 BF16 = **block 级保精度**；[2606.12280](https://arxiv.org/abs/2606.12280) "高脆弱层集" bf16 保护 = **层集级保精度**。三者同一原则的不同粒度，可叠加。

---

## 8. 发散点 → 机理 → 阈值/条件 → 修复策略 → 适用设备（汇总表）

| 发散点 | 机理 | 阈值/条件 | 修复策略 | 适用设备 |
|---|---|---|---|---|
| Softmax/attention | 大序列 `x-max` 减法 NaN/下溢；fp16 exp 下溢 | seq≥万级；score差>11 下溢 | fp32 softmax；flash-attn fp32 路径 | MPS(#96602)、消费NV fp16、NPU |
| GELU-tanh | Metal 融合内核缺陷；fp16 `x³` 溢出 65504 | bf16 \|x\|≥15 NaN；fp16 \|x\|>40 溢出 | cast fp32 计算（你的方案） | MPS、CoreML fp16、NPU |
| AdaLN 调制 (1+scale) | bf16 7位尾数灾难性抵消 | \|1+scale\|<2^-7≈0.0078 | 调制参数+乘加 fp32（你的方案）；训练期 scale 远离 -1 正则 | 全低精度后端（bf16/int8/int4/NPU） |
| RMSNorm/LayerNorm | 平方和溢出/方差精度差；MPS fp16 崩溃 | fp16 \|x\|>256 平方溢出 | 统计与除法 fp32 | MPS(#96113)、fp16 CoreML、NPU |
| VAE 解码 | 大数值范围、GroupNorm 不稳 | 中间特征 O(100) | 全程 fp16/fp32，不 int8 | 全后端 |
| 时间步嵌入 | 频率展开量级跨度大→喂 AdaLN | timestep 静态固化(ONNX #11456) | MLP fp32；动态 shape | ONNX、CoreML、MPS |
| embedding 大表 | 源>2³¹ 静默全零 | >2³¹ 元素 | 分片/限规模 | MPS(#188756) |
| baddbmm/线性代数 | beta==0 错误传播 NaN | beta==0 | 显式 beta≠0 或 fp32 | MPS(#187521) |
| multinomial/Gumbel | exponential_() 产生 -0.0→NaN | 零概率索引 | fp32 概率或避免 0 概率 | MPS(#192577) |
| 量化激活 outlier | int4/int8 表不下 outlier 通道 | outlier 通道幅度 >>均值 | SmoothQuant/SVDQuant/Hadamard 旋转；per-channel | 消费NV int8、NPU int4 |
| timestep 漂移敏感 | 激活随 timestep 强变 | 高噪声步激活 outlier 多 | 时间步感知混合精度(AdaTSQ/6Bit-Diffusion) | 全量化后端 |
| fp8→int8 格式错配 | fp8 范围假设≠int8 | E4M3 max≈448 | Hadamard 旋转预处理(HyperQuant: int8 击败 fp8) | 消费NV、NPU |

---

## 9. 桥接结论：训练侧如何让模型天然适配异构端侧低精度推理

1. **训练期模拟端侧混合精度（fake-quant QAT）**：前向插 `quant_dequant` 模拟目标端侧的"int8/int4 权重 + bf16 激活 + 关键 op fp32"混合精度，反向用 STE。用 [EfficientDM](https://arxiv.org/abs/2310.03270) 的 QALoRA + 蒸馏范式，或 [QVGen](https://arxiv.org/abs/2505.11497)（视频）的辅助模块+rank-decay。**用端侧真实精度配置模拟**，而非通用 fake-quant，使训练分布与推理一致（[2412.06661](https://arxiv.org/abs/2412.06661) 的训练-推理一致性原则）。

2. **AdaLN 调制参数加"远离精度悬崖"正则**：在损失加 `λ · mean(relu(τ - |1+scale|))`（τ≈0.05）约束 scale 不进入 bf16 抵消区（|1+scale|<2^-7）。这是你的灾难性抵消发现的训练侧根因消除——**未见专门论文做此正则，是可发表的研究点**，与推理期 boundary-block 保高精度（[2606.00957](https://arxiv.org/abs/2606.00957)）互补。

3. **训练时用 Hadamard 旋转预处理权重/激活**：让模型适应旋转后近高斯分布，下游 int8/int4/fp8 任何格式都更稳（[OrbitQuant](https://arxiv.org/abs/2607.02461)、[HQ-DM](https://arxiv.org/abs/2512.05746)、[HyperQuant](https://arxiv.org/abs/2606.23406) "int8 击败 fp8"）。可在训练期固定旋转并让模型收敛到旋转友好权重。

4. **蒸馏到量化友好的小模型**：与其量化 14B 大 DiT，不如蒸馏到 1–2B 学生再量化（[2605.16259](https://arxiv.org/abs/2605.16259) SDXS + CoreML 达实时）。学生模型激活分布更窄、outlier 更少，端侧低精度更友好。

5. **fp8 训练需匹配端侧导出分档 + 重要性采样校正**：若用 fp8 训练（[FPSAttention](https://arxiv.org/abs/2506.04648)），须用 [AIS 2605.13907](https://arxiv.org/abs/2605.13907) 的重要性采样校正 FP8-BF16 不匹配，否则权重带偏置；导出时按端侧能力分档（Blackwell→NVFP4/fp8；消费 Ampere→INT8 fused；MPS/NPU→int8/int4 + §7 模板），并为 MPS/NPU 准备独立的 int8 + bf16 关键层变体。

---

## Sources（2025–2026 重点）

### arXiv 论文（DiT 量化 / 数值鲁棒 / 端侧部署）
- [KroQuant: Kronecker-Structured Block Transforms for DiT PTQ (2026-07)](https://arxiv.org/abs/2607.21446)
- [OrbitQuant: Data-Agnostic Quantization for Image and Video DiTs (2026-07)](https://arxiv.org/abs/2607.02461)
- [HyperQuant: LTX-2 19B 视频 DiT 近无损量化 (2026-06)](https://arxiv.org/abs/2606.23406)
- [Boundary-Protection W8A8 HiFloat8 for Wan2.1-T2V-14B on Ascend 910B NPU (2026-05, ICME 2026)](https://arxiv.org/abs/2606.00957)
- [Realizing Native INT8 Compute for DiT on Consumer GPUs — Ideogram 4.0 (2026-06)](https://arxiv.org/abs/2606.14598)
- [Holding the FP8 Quality Ceiling — Ideogram 4.0 INT8+GGUF, PSNR/SSIM/LPIPS (2026-06)](https://arxiv.org/abs/2606.12280)
- [Timestep-Aware SVDQuant-GPTQ for Wan2.2-I2V W4A4 (2026-05)](https://arxiv.org/abs/2605.27003)
- [DiRotQ: Rotation-Aware W4A4 for DiT (2026-05)](https://arxiv.org/abs/2605.16732)
- [Q-ARVD: Quantizing Autoregressive Video Diffusion (2026-05)](https://arxiv.org/abs/2605.21072)
- [6Bit-Diffusion: NVFP4/INT8 Mixed-Precision Video DiT (2026-03)](https://arxiv.org/abs/2603.18742)
- [SemanticDialect: Mixed-Format Quantization for Video DiT (2026-03)](https://arxiv.org/abs/2603.02883)
- [AdaTSQ: Temporal-Sensitivity DiT Quantization (2026-02)](https://arxiv.org/abs/2602.09883)
- [Q-DiT4SR: Hierarchical SVD DiT Quantization (2026-02, ICML 2026)](https://arxiv.org/abs/2602.01273)
- [Attn-QAT: 4-Bit Attention QAT, FP4 训练不稳定 (2026-02, upd 2026-08)](https://arxiv.org/abs/2603.00040)
- [BinaryAttention: 1-Bit QK-Attention (CVPR 2026)](https://arxiv.org/abs/2603.09582)
- [TreeQ: Near-Lossless 4-bit DiT PTQ (2025-12)](https://arxiv.org/abs/2512.06353)
- [HQ-DM: Single Hadamard QAT for Diffusion (2025-12, upd 2026-08)](https://arxiv.org/abs/2512.05746)
- [RobuQ: W1.58A2 DiT QAT (ICML 2026)](https://arxiv.org/abs/2509.23582)
- [QVGen: QAT for Video Generative Models (ICLR 2026)](https://arxiv.org/abs/2505.11497)
- [FPSAttention: Training-Aware FP8+Sparsity for Wan2.1 Video Diffusion (2025-06)](https://arxiv.org/abs/2506.04648)
- [ViDiT-Q: DiT Quantization, W8A8/W4A8 (ICLR 2025)](https://arxiv.org/abs/2406.02540)
- [DilateQuant: Weight Dilation QAT (ACMMM 2025)](https://arxiv.org/abs/2409.14307)
- [Efficiency Meets Fidelity: 训练-推理一致性 SD 量化 (2024-12)](https://arxiv.org/abs/2412.06661)
- [SVDQuant: 4-bit Low-Rank Outlier Absorption (ICLR 2025 Spotlight)](https://arxiv.org/abs/2411.05007)
- [EfficientDM: QALoRA Data-Free QAT (ICLR 2024)](https://arxiv.org/abs/2310.03270)
- [SmoothQuant: W8A8 Outlier Migration (ICML 2023)](https://arxiv.org/abs/2211.10438)
- [Efficient FP8 PTQ: E4M3/E5M2/E3M4 对比 (Intel)](https://arxiv.org/abs/2309.14592)
- [AIS: FP8 rollout vs BF16 trainer 不匹配 (2026-05)](https://arxiv.org/abs/2605.13907)
- [LaCache: Per-group FP8 for DLLM (2026-07)](https://arxiv.org/abs/2607.16339)
- [STaR-Quant: State-Time Consistent PTQ (2026-06)](https://arxiv.org/abs/2606.04945)
- [LSGQuant: Layer-Sensitivity Video SR (2026-02)](https://arxiv.org/abs/2602.03182)

### Apple Silicon / MLX / CoreML / ANE
- [Systematic Optimization of Real-Time Diffusion on Apple M3 Ultra — CoreML/量化/ANE (2026-02)](https://arxiv.org/abs/2605.16259)
- [ANEForge: Direct ANE Programming w/o CoreML, int8/int4/sparse (2026-06)](https://arxiv.org/abs/2606.17090)
- [Production-Grade Local LLM on Apple Silicon: MLX vs MPS vs llama.cpp (2025-10)](https://arxiv.org/abs/2511.05502)
- [Benchmarking On-Device ML on Apple Silicon with MLX (2025-10)](https://arxiv.org/abs/2510.18921)
- [coremltools #1763: ANE MHA Export Fails with Flexible Shape](https://github.com/apple/coremltools/issues/1763)
- [coremltools #1359: Einsum unsupported equation](https://github.com/apple/coremltools/issues/1359)
- [coremltools #1947: Float16 MLMultiArray not supported](https://github.com/apple/coremltools/issues/1947)

### PyTorch MPS 数值缺陷（issue 追踪）
- [#96602 softmax 大张量 NaN (Open, 2023)](https://github.com/pytorch/pytorch/issues/96602)
- [#96113 LayerNorm fp16 崩溃 (Closed 2.0.1)](https://github.com/pytorch/pytorch/issues/96113)
- [#192577 multinomial/Gumbel NaN](https://github.com/pytorch/pytorch/issues/192577)
- [#192507 hypot NaN for inf](https://github.com/pytorch/pytorch/issues/192507)
- [#190476 nextafter no-op for bf16](https://github.com/pytorch/pytorch/issues/190476)
- [#190057 LSTM dropout=1.0 all NaN](https://github.com/pytorch/pytorch/issues/190057)
- [#189243 div floor nonfinite semantics](https://github.com/pytorch/pytorch/issues/189243)
- [#188756 embedding 全零 >2³¹](https://github.com/pytorch/pytorch/issues/188756)
- [#187521 baddbmm NaN when beta==0](https://github.com/pytorch/pytorch/issues/187521)
- [#190632 MPS 非确定性 NaN/Inf 梯度](https://github.com/pytorch/pytorch/issues/190632)
- [#187455 MPSGraph→原生 Metal 迁移](https://github.com/pytorch/pytorch/issues/187455)

### ComfyUI / 转换管线
- [ComfyUI #15315: MiniMax H3 T2V 全黑视频+NaN音频 on M4 Max (Closed 2026-08-10)](https://github.com/Comfy-Org/ComfyUI/issues/15315)
- [city96/ComfyUI-GGUF README — DiT 量化友好, TE/VAE 分离, MacOS torch 2.4.1](https://github.com/city96/ComfyUI-GGUF)
- [diffusers #11456: ONNX export timestep 静态值失败 (Open)](https://github.com/huggingface/diffusers/issues/11456)
- [diffusers #11032: TRT quantization backend (Closed)](https://github.com/huggingface/diffusers/issues/11032)
- [diffusers #6336: TensorRT VAE 编译报错](https://github.com/huggingface/diffusers/issues/6336)
- [diffusers #7922: ORT SDXL pipeline](https://github.com/huggingface/diffusers/issues/7922)
- [ONNX GraphSurgeon 拆分单体扩散 planner (2026-03)](https://arxiv.org/abs/2603.01023)
