# 端侧视频生成 DiT 模型运行时与部署栈调研（2025–2026）

> 调研日期：2026-08-13
> 范围：训练在 NVIDIA 数据中心（H100/H200/B200/GB300），本报告专注**端侧推理运行时栈**，分 A/B/C 三类设备。
>
> **数据采集说明（重要）**：本次调研环境的 `web_search`（Brave）API key 未配置，无法做搜索引擎发现。所有结论均通过 **`web_fetch` 抓取一手来源**（GitHub 仓库 README/源码、NVIDIA/Apple 官方产品页、MLX 官方文档 v0.32）+ **持久浏览器**（读取 NVIDIA Jetson Thor 规格表）获得，文末附 URL。未通过搜索发现的近期 benchmark/博客可能覆盖不全；凡未在来源中直接看到的数字（如部分前代芯片规格、带宽推导值）均明确标注「常见公开规格 / 推导值」，不冒充已抓取。

---

## 0. 关键结论速览

- **A 类 Apple Silicon**：MLX 已官方跑通 **Wan2.1 T2V/I2V**，并给出 M4 Max 真实延迟（1.3B ~90 s/it，14B ~230 s/it，81 帧）。统一内存让 14B 模型可直接常驻（~36GB），但**内存带宽（M3 Ultra 819 GB/s）远低于离散 GPU**，是吞吐瓶颈。MLX v0.32 已支持 FP8（`to_fp8`/`from_fp8`）、原生分组量化、GGUF 保存、SDPA fast 路径、Metal+CUDA 双后端、分布式/张量并行。
- **B 类 消费级 NVIDIA**：ComfyUI 已内置**原生 FP8 量化**（`QuantizedTensor` + 逐层混合精度 + PTQ 校准），不止 GGUF。RTX 5090 Blackwell 的 **FP4/NVFP4** 是差异化点：SageAttention3 用 microscaling FP4，LightX2V 的 Wan2.2-NVFP4-Sparse 在单卡 RTX 5090 上 **>50× 加速**。
- **C 类 工业边缘/NPU**：**Jetson Thor**（Blackwell 边缘）已发布，AGX Thor/T5000 达 **2070 TFLOPS FP4-sparse、128GB LPDDR5X 统一内存、273 GB/s、40–130W**；国产/移动 NPU 的视频 DiT 主要靠 **LightX2V**（已适配 Ascend 910B、寒武纪 MLU590、MetaX、摩尔线程、海光、燧原、天数、T-head、Intel AIPC）与 ComfyUI（支持 Ascend/Cambricon/Iluvitar）落地，低端 NPU（RK3588 6 TOPS）不可行，需极端蒸馏+量化。

---

## A. Apple Silicon（M 系列，MPS）

### A.1 运行时栈

| 组件 | 现状（2025–2026） | 来源 |
|---|---|---|
| **MLX**（Apple 官方） | v0.32。统一内存模型、惰性求值、动态图、Python/C++/Swift API。支持 fp16/bf16/**fp8**（`to_fp8`/`from_fp8`）、原生量化（`quantize`/`quantized_matmul`/`QQLinear`/`nn.QuantizedLinear`）、`save_gguf`/`save_safetensors`、`fast.scaled_dot_product_attention`、自定义 `metal_kernel` **与** `cuda_kernel`、分布式通信（all_gather/sum_scatter/send/recv）、张量并行、`set_wired_limit`/`set_cache_limit`。后端：Metal GPU + CPU（另有 Linux CUDA/CPU 包）。 | [MLX repo](https://github.com/ml-explore/mlx)、[MLX 0.32 文档](https://ml-explore.github.io/mlx/build/html/index.html) |
| **PyTorch MPS backend** | ComfyUI 在 Apple Silicon（M1–M4）上经 PyTorch nightly MPS 运行；`--preview-method`、模型 offload 等。已知 MPS 内核存在数值坑（bf16 GELU/AdaLN 精度、部分算子 CPU 回退）；ComfyUI-GGUF 注明 macOS Sequoia 需 torch 2.4.1（2.6 nightly 触发 "M1 buffer is not large enough"）。 | [ComfyUI README](https://github.com/comfyanonymous/ComfyUI)、[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) |
| **CoreML + ANE** | M3 Ultra 32 核 Neural Engine、M4 Max 16 核；硬件 ProRes/AV1 media engine。ANE 对 conv/linear INT8 强，但 **DiT transformer 算子覆盖率低**，CoreML 非视频 DiT 的一等公民（多用于图像/LLM/Whisper）。 | [Apple Mac Studio specs](https://www.apple.com/mac-studio/specs/) |
| **MPSGraph / Metal** | 底层；MLX 构建于 Metal 之上，可写 `metal_kernel` 自定义算子。 | MLX 文档 |

> 注：截至抓取日期，**Apple 尚未发布 M4 Ultra**；当前 Mac Studio M4 世代顶配为 **M4 Max 或 M3 Ultra**。M2 Ultra 为上一代（Mac Pro/Studio）。

### A.2 量化路径

1. **MLX 原生分组量化**：`--quantize`（Wan2.1 示例）走 `quantize`/`QuantizedLinear`，4/8-bit 分组；`--no-cache` 可再省内存（1.3B 480p 81 帧：~10GB vs ~14GB）。 [MLX Wan2.1 README](https://github.com/ml-explore/mlx-examples/blob/main/video/wan2.1/README.md)
2. **GGUF**：`ComfyUI-GGUF`（city96）——关键结论「**DiT/transformer 模型对量化不敏感，可用更低 bpw 可变比特量化；conv2d UNet 则不适合**」。预量化模型含 Flux/SD3.5/T5。llama.cpp Metal 后端亦支持扩散模型。MLX 本身可 `save_gguf`。 [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
3. **FP8**：MLX `to_fp8`/`from_fp8`（较新）；Apple GPU 无 Blackwell 式 FP4 张量核，FP4 需软件/Metal 实现。
4. **步数蒸馏**：MLX Wan2.1 示例直接支持 lightx2v **4 步蒸馏** Wan2.1 14B（`--checkpoint ...4step.safetensors --sampler euler --steps 4 --guidance 1.0`），跳过 CFG。 [lightx2v Wan2.1-Distill-Models](https://huggingface.co/lightx2v/Wan2.1-Distill-Models)
5. **TeaCache**：跳过 20–60% 冗余前向（阈值 0.05→~34% 跳过近乎无损；0.1→~58%；0.25→~76%）。 [TeaCache arXiv 2411.19108](https://arxiv.org/abs/2411.19108)

### A.3 能力边界与坑

- **统一内存优势**：无 VRAM/RAM 分割，14B（~36GB 未量化）可常驻于 64GB+ Mac；`set_wired_limit` 提高可用 wired 内存。
- **带宽是瓶颈**：M3 Ultra 819 GB/s vs RTX 5090 ~1.79 TB/s vs H100 ~3.35 TB/s。注意力为 memory-bound，单步延迟显著高于离散 GPU（见下表 M4 Max 90–250 s/it）。
- **MPS 数值坑**：bf16 GELU/AdaLN 精度问题、部分算子 CPU 回退、torch 版本钉死（Sequoia 需 2.4.1）。
- **ANE 覆盖低**：DiT 的 3D 注意力/AdaLN/RoPE 在 ANE 上支持差，CoreML 路径目前不主流。
- **无 FP4 硬件**：Apple GPU 不具备 Blackwell FP4 张量核；最高 FP8（软件/Metal）。
- **功耗友好**：Mac Studio 系统最大连续功率 480W（M3 Ultra 实际远低于此）。

### A.4 典型硬件规格

| 设备 | GPU | Neural Engine | 内存带宽 | 统一内存上限 | 媒体引擎 | 系统/功耗 |
|---|---|---|---|---|---|---|
| **M4 Max**（Mac Studio） | 32–40 核 | 16 核 | **410–546 GB/s** | 64GB | 2× 视频编码、2× ProRes、AV1 解码 | Mac Studio 最大 480W |
| **M3 Ultra**（Mac Studio） | 60–80 核 | 32 核 | **819 GB/s** | 最高 512GB | 4× 视频编码、4× ProRes、AV1 解码 | Mac Studio 最大 480W |
| M2 Ultra（上代，常见公开规格） | 60–76 核 | 32 核 | 800 GB/s | 最高 192GB | ProRes | ~370W |

来源：[Apple Mac Studio 技术规格](https://www.apple.com/mac-studio/specs/)（M4 Max/M3 Ultra 带宽与媒体引擎为一手抓取；M2 Ultra 为常见公开规格，未单独抓取）。

### A.5 视频生成落地案例与延迟

**MLX 官方 Wan2.1，M4 Max 芯片，81 帧**（一手数据）：

| 模型 | 任务 | 未量化 RAM | 单 DiT 步 (M4 Max) | 50 步粗估 | 4 步蒸馏+量化粗估 |
|---|---|---|---|---|---|
| Wan2.1-T2V-1.3B | T2V | ~10GB | **~90 s/it** | ~75 min | ~6 min + VAE |
| Wan2.1-T2V-14B | T2V | ~36GB | **~230 s/it** | ~3.2 hr | ~15 min + VAE |
| Wan2.1-I2V-14B-480P | I2V | ~39GB | ~250 s/it | ~3.5 hr | ~17 min + VAE |

- 配 `--quantize` + 4 步蒸馏（lightx2v）+ TeaCache(0.05) 后，1.3B 可压到数分钟级；14B 仍需 ~15 分钟级。
- **HunyuanVideo / LTX-Video 暂无 MLX 原生移植**（mlx-examples 仅 Wan2.1）；LTX-Video 在 ComfyUI 跨平台支持，理论上可走 MPS，但无官方 MLX 路径。

来源：[MLX Wan2.1 README](https://github.com/ml-explore/mlx-examples/blob/main/video/wan2.1/README.md)

---

## B. 消费级 NVIDIA GPU（RTX 4090 / 5090 / RTX 6000 Ada / Blackwell）

### B.1 运行时栈

| 组件 | 现状 | 来源 |
|---|---|---|
| **TensorRT / TRT-LLM for diffusion** | NVIDIA 优化引擎，FP8/INT8/FP4 内核、CUDA graphs。 | [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) |
| **torch.compile（inductor, max-autotune）+ CUDA graphs** | ComfyUI/LightX2V 常用；SageAttention 支持 `torch.compile`（非 cudagraphs 模式）。 | [SageAttention](https://github.com/thu-ml/SageAttention) |
| **cuDNN / xformers / FlashAttention2-3** | 基线注意力；FA3-FP8 在 Hopper 上与 SageAttention 速度相当但精度更低。 | SageAttention README |
| **SageAttention 1/2/2++/3** | 量化注意力，比 FA 快 2–5×、端到端无损（语言/图像/视频）。**SageAttention3 = microscaling FP4（Blackwell）**；对图像/视频模型「**只替换 DiT 内的 attention**」。ICLR2025/ICML2025/NeurIPS2025 Spotlight。 | [SageAttention](https://github.com/thu-ml/SageAttention) |
| **ComfyUI 原生**（此类设备最成熟） | 内置 `QuantizedTensor`（torch.Tensor 子类）+ 两级注册 + 逐层混合精度（`MixedPrecisionOps`）+ **PTQ 校准**（activation `input_scale`）。FP8 `float8_e4m3fn` 存于 safetensors 带 `_quantization_metadata`。 | [ComfyUI QUANTIZATION.md](https://github.com/Comfy-Org/ComfyUI/blob/master/QUANTIZATION.md) |
| **LightX2V**（ModelTC） | 4 步蒸馏、NVFP4/FP8/INT8 量化、disk-CPU-GPU 三级 offload、TeaCache/MagCache、集成 Sage/Flash/Radial Attention、sgl-kernel/vllm。 | [LightX2V](https://github.com/ModelTC/LightX2V) |

### B.2 量化路径

- **Blackwell（RTX 5090）**：**FP4/NVFP4** 原生张量核（页面明确 "Max AI performance with FP4 and DLSS 4.5"）、FP8（e4m3）、FP16/BF16。LightX2V Wan2.2-NVFP4-Sparse 在单卡 5090 上 **>50× 加速**；SageAttention3 在 RTX5090 达 **560T，比 FA2 快 2.7×**（2025-02-15 发布）。CUDA ≥12.8。
- **Ada（RTX 4090 / 6000 Ada）**：FP8（e4m3，Ada 张量核，CUDA ≥12.4）、FP16/BF16、INT8。无 FP4。
- **GGUF**：可用（ComfyUI-GGUF），但在 NVIDIA 上**多用原生 FP8/FP4**，GGUF 主要为低 VRAM 兜底。

### B.3 显存分层与模型选型

- **24GB（4090）**：LightX2V 让 HunyuanVideo-1.5 可在 24GB 4090 跑（同 GPU 数下 >2× 提速）；Wan 14B 走 offload/GGUF。**极限：8GB VRAM + 16GB RAM 即可跑 14B 480P/720P**（三级 offload）。
- **32GB（5090）**：Wan2.2 14B 舒适，FP8/NVFP4，720P。
- **48GB（RTX 6000 Ada）**：更大 batch、720P/1080P、多流。

### B.4 能力边界与坑

- FP4 质量需 NVFP4 量化感知蒸馏（LightX2V 的 QAT-step-distill），朴素 PTQ FP4 有精度损失。
- 24GB 跑全精度 14B 会 OOM（xDiT/SGL 在 4090D 单卡 OOM，见下表）；必须蒸馏+量化+offload。
- 5090 功耗高（575W TGP，需 1000W 系统、4×8-pin 或 1×600W PCIe Gen5 线），热与电源是部署约束。

### B.5 典型硬件规格

| GPU | 架构 | VRAM | 位宽/带宽 | AI 算力 | FP8 | FP4 | TGP |
|---|---|---|---|---|---|---|---|
| **RTX 5090** | Blackwell | **32GB GDDR7** | 512-bit / ~1.79 TB/s（28 Gbps×512/8，推导值） | **3352 AI TOPS（FP4-sparse）**；21,760 CUDA；2.41GHz boost | ✅ e4m3 | ✅ NVFP4 | **575W**（系统 1000W） |
| RTX 4090 | Ada Lovelace | 24GB GDDR6X | 384-bit / 1008 GB/s | 4th-gen 张量核；82.6 TFLOPS FP32（常见公开规格） | ✅ e4m3 | ❌ | 450W |
| RTX 6000 Ada | Ada Lovelace | 48GB GDDR6 | 384-bit / 960 GB/s | 4th-gen 张量核（常见公开规格） | ✅ e4m3 | ❌ | 300W |

来源：[NVIDIA RTX 5090 产品页](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)（一手：32GB/512-bit/3352 FP4 TOPS/575W/FP4+DLSS4.5）；4090 与 6000 Ada 规格为常见公开规格（带宽 5090 为推导值，明确标注）。

### B.6 视频生成落地案例与延迟

**LightX2V，Wan2.1-I2V-14B-480P，40 步，81 帧**（一手 benchmark，2025-12-01 更新）：

| GPU | 配置 | 单步 | 40 步粗估 | 备注 |
|---|---|---|---|---|
| RTX 4090D ×1 | cfg | **20.26 s/it** | ~13.5 min | Diffusers 30.5s/it；xDiT/SGL OOM |
| RTX 4090D ×8 | cfg | 4.75 s/it | ~3.2 min | no-cfg 3.13；fp8 2.35 |
| H100 ×1 | cfg | 5.18 s/it | ~3.5 min | Diffusers 9.77s/it |
| H100 ×8 | cfg | 0.75 s/it | ~30 s | no-cfg 0.39；fp8 0.35 |

- **4 步蒸馏**：单卡 ~20× 加速（LightX2V）；HunyuanVideo-1.5 4 步蒸馏 ~25× vs 50 步。
- **Wan2.2-NVFP4-Sparse（Blackwell FP4）**：单卡 RTX 5090 **>50× 加速**（2026-05-29 发布）。
- 功耗：5090 575W / 4090 450W / 6000 Ada 300W。

来源：[LightX2V README](https://github.com/ModelTC/LightX2V)

---

## C. 工业边缘 / NPU

### C.1 运行时栈

#### C.1.a NVIDIA Jetson Thor（Blackwell 边缘，2025–2026 重点）
- **TensorRT**（FP4/FP8/INT8）、CUDA、第五代张量核、**MIG**（多实例 GPU，10/6 TPC）、PVA v3 视觉加速器、**2× NVENC / 2× NVDEC**（T4000 为 1×）。
- JetPack SDK、Holoscan、Isaac（GR00T N1 机器人栈）。
- **统一 LPDDR5X 内存**（128GB/64GB），非 VRAM/RAM 分割。

#### C.1.b Jetson Orin（上一代边缘）
- TensorRT INT8、**DLA（NVDLA）×2**、NVENC/NVDEC、JetPack。INT8 PTQ 校准/QAT 流程成熟。

#### C.1.c 非 NVIDIA NPU
- **国产/移动 NPU 厂商 SDK**：Rockchip RKNN、华为 CANN/`torch_npu`、高通 QNN/SNPE、联发科 NeuronPilot；模型转换 ONNX → 厂商编译器。
- **ComfyUI 已支持** Ascend（`torch_npu`）、寒武纪（`torch_mlu`）、天数 Iluvitar（CoreX）。 [ComfyUI README](https://github.com/comfyanonymous/ComfyUI)
- **LightX2V 已适配**：Ascend 910B、寒武纪 MLU590、MetaX C500、摩尔线程 MUSA、海光 DCU、燧原 S60、天数 iluvatar、T-head PPU、Intel AIPC PTL；**GGUF 推理已在 Cambricon MLU590 / MetaX C500 落地**。 [LightX2V README](https://github.com/ModelTC/LightX2V)

### C.2 量化路径

- **Jetson Thor**：**FP4/NVFP4 原生**、FP8、INT8 PTQ/QAT（TensorRT）。2070 TFLOPS FP4-sparse。
- **Jetson Orin**：INT8 PTQ 校准（TensorRTEntropy/Percentile）、QAT、DLA INT8。
- **非 NVIDIA**：INT8（w8a8）为主，部分支持 INT4/NVFP4（LightX2V w4a4-nvfp4）；厂商量化工具。**DiT 算子覆盖缺口**（3D 注意力、AdaLN、RoPE）常需自定义算子或回退。

### C.3 能力边界与坑

- **Jetson Thor**：128GB 统一内存可常驻 14B（FP8/NVFP4），但 **273 GB/s 带宽远低于离散 GPU**，注意力 memory-bound；40–130W TDP、被动/主动散热约束持续吞吐。
- **DLA 限制**：Orin DLA 算子集窄，DiT 多数层仍走 GPU；Thor 以 GPU+PVA 为主。
- **非 NVIDIA NPU**：低端（RK3588 6 TOPS）视频 DiT 不可行；移动 NPU 需极端 4 步蒸馏+INT8/INT4+低分辨率，算子覆盖缺口大；**服务器级国产卡**（Ascend 910B / 寒武纪 MLU590）经 LightX2V/ComfyUI 可行。
- **Jetson Thor 视频生成公开延迟**：截至抓取日期未见公开 benchmark；下表延迟为基于 4090D（~20 s/it）与 Thor 带宽/算力比值的**定性外推**（带宽约为 4090 的 1/4，FP4 张量吞吐高），预期 480p 短 clip 为**多分钟级、带宽受限**——明确标注未公开实测。

### C.4 典型硬件规格

| 设备 | 算力 | 内存 | 带宽 | TDP | 量化 |
|---|---|---|---|---|---|
| **Jetson AGX Thor / T5000** | **2070 TFLOPS FP4-sparse**（2560 核 Blackwell, 5th-gen TC, MIG 10 TPC, 1.57GHz） | **128GB LPDDR5X** | **273 GB/s**（256-bit） | **40–130W** | FP4/FP8/INT8 |
| **Jetson T4000** | 1200 TFLOPS FP4-sparse（1536 核, 1.53GHz） | 64GB LPDDR5X | 273 GB/s | 40–70W | FP4/FP8/INT8 |
| Jetson AGX Orin 64GB（常见公开规格） | 275 TOPS INT8-sparse（Ampere 2048 核） | 64GB LPDDR5 | 204.8 GB/s | 15–60W | INT8/FP16 |
| Rockchip RK3588（Rockchip 数据手册公开规格） | 6 TOPS INT8 NPU（3 核） | ≤32GB LPDDR4x | ~50 GB/s | ~5–15W | INT8/INT4 |
| Qualcomm Hexagon（Snapdragon 8 Gen 3，厂商公开） | ~45 TOPS NPU | LPDDR5 | 高 | 移动 TDP | INT8/INT4 |
| MediaTek Dimensity APU（厂商公开） | ~35 TOPS | LPDDR5X | 高 | 移动 TDP | INT8 |
| 华为 Ascend 310（边缘，厂商公开） | 22 TOPS INT8 | LPDDR4 | ~51 GB/s | 8W | INT8 |
| Ascend 910B（经 LightX2V 适配，服务器级） | ~数百 TFLOPS FP16 | 32–64GB | 高带宽 | ~310W | INT8/FP16 |

来源：[NVIDIA Jetson Thor 产品页](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/)（一手：2070/1200 TFLOPS FP4、128/64GB、273 GB/s、40–130W/40–70W、2560/1536 核、NVENC×2）；Orin/RK3588/Hexagon/APU/Ascend 为厂商公开规格（标注），未单独抓取。

### C.5 视频生成落地案例

- **Jetson Thor**：NVIDIA 主推物理 AI/机器人（Isaac GR00T N1、VSS、Holoscan）；视频 DiT 可经 TensorRT FP4/INT8 + 4 步蒸馏运行，但**无公开端到端延迟**。
- **国产加速卡**：LightX2V 在 Ascend 910B / 寒武纪 MLU590 / MetaX / 摩尔线程 / 海光 / 燧原 / 天数 上跑 Wan/HunyuanVideo/LTX/MiniMax-H3（含 4 步蒸馏、FP8/NVFP4、GGUF）——这是国产卡跑视频 DiT 的**主路径**。
- **RK3588 / 移动 NPU**：仅适合极端蒸馏的微型图像/超低分辨率视频；全量视频 DiT 不可行。

来源：[LightX2V README](https://github.com/ModelTC/LightX2V)、[ComfyUI README](https://github.com/comfyanonymous/ComfyUI)

---

## 总表：三类端侧设备对比

| 维度 | A. Apple Silicon | B. 消费级 NVIDIA | C. 工业边缘/NPU |
|---|---|---|---|
| **显存/内存** | 统一内存，M3 Ultra 最高 512GB / M4 Max 64GB | 独立 VRAM：24/32/48GB | Jetson Thor 128/64GB 统一；RK3588 ≤32GB；移动 NPU 共享 LPDDR |
| **峰值算力** | M3 Ultra 819 GB/s（带宽主导，无公开 TFLOPS 口径） | RTX 5090 **3352 AI TOPS FP4**；4090 FP8 | Thor **2070 TFLOPS FP4-sparse**；Orin 275 TOPS INT8；RK3588 6 TOPS |
| **量化精度** | FP16/BF16/FP8（MLX）、4/8-bit 分组、GGUF；**无 FP4 硬件** | FP4/NVFP4（Blackwell）、FP8、INT8、GGUF | FP4/FP8/INT8（Thor）；INT8/INT4（非 NVIDIA NPU） |
| **视频生成可达性** | Wan2.1 官方跑通（MLX）；Hunyuan/LTX 无原生移植 | 最成熟（ComfyUI 原生 + LightX2V），Wan/Hunyuan/LTX/CogVideoX/MiniMax 全覆盖 | Thor 可行但无公开延迟；国产卡经 LightX2V 可行；低端 NPU 不可行 |
| **可达延迟（480p 81 帧）** | M4 Max：1.3B ~75 min（50 步）/ 数分钟（4 步蒸馏+量化）；14B ~3 hr | 4090D ~13.5 min（40 步）/ ~1 min（4 步蒸馏）；5090 NVFP4 >50× 加速 | Thor 多分钟级（外推，带宽受限）；H100×8 ~30 s |
| **主要瓶颈** | 内存带宽（819 GB/s）、MPS 数值坑、ANE 覆盖低 | 24GB 显存上限、5090 功耗/热 | 带宽（273 GB/s）、功耗/散热、非 NVIDIA 算子覆盖缺口 |
| **推荐模型规模** | 1.3B 常态 / 14B 仅大内存+蒸馏 | 14B（蒸馏+FP8/NVFP4）；24GB 走 offload | Thor：14B NVFP4 4 步蒸馏；RK3588：不可行 / 仅微型 |
| **功耗** | Mac Studio ≤480W 系统 | 575W / 450W / 300W | 40–130W（Thor）/ 15–60W（Orin）/ 5–15W（RK3588） |

---

## 按端侧类型选模型的决策建议

1. **统一内存 ≥128GB（Mac M3 Ultra 512GB / Jetson Thor 128GB）**：可常驻 14B bf16/fp8 **无需 offload**，但带宽（273–819 GB/s）是吞吐瓶颈 → **必上 4 步蒸馏 + TeaCache** 削减步数，接受较低吞吐。Mac M3 Ultra 适合大模型原型/低并发长任务；Jetson Thor 适合边缘离线/批处理短 clip。
2. **24GB 独立显存（RTX 4090）**：**蒸馏 14B + FP8/GGUF Q4 + offload**（LightX2V 8GB+16GB 路径）跑 Wan/HunyuanVideo 480P；基线 ~13 min/clip，4 步蒸馏可到 ~1 min。避免全精度 14B（OOM）。
3. **32GB Blackwell（RTX 5090）**：优先 **NVFP4 + SageAttention3 + 4 步蒸馏**（Wan2.2-NVFP4-Sparse >50×）；720P 可达。FP4 是相对 Ada 的核心差异化，务必走量化感知蒸馏而非朴素 PTQ。
4. **Jetson Thor 边缘**：**NVFP4/FP4 + INT8 + 4 步蒸馏**；128GB 统一内存可装 14B，但 273 GB/s 带宽限制吞吐 → 目标短低分辨率 clip、离线或低频批处理；注意 40–130W 持续散热降额。
5. **非 NVIDIA NPU**：低端（RK3588 6 TOPS / 移动 Hexagon/APU）视频 DiT **不可行**，需极端 4 步蒸馏 + INT8/INT4 + 大幅降分辨率，且算子覆盖缺口需自定义算子/回退。国产**服务器级卡**（Ascend 910B / 寒武纪 MLU590 / MetaX）经 **LightX2V 或 ComfyUI（torch_npu/torch_mlu）** 是落地视频 DiT 的现实主路径。

---

## 来源汇总

- MLX 框架：https://github.com/ml-explore/mlx
- MLX 文档 v0.32：https://ml-explore.github.io/mlx/build/html/index.html
- MLX Wan2.1 示例（M4 Max 延迟/量化/蒸馏/TeaCache）：https://github.com/ml-explore/mlx-examples/blob/main/video/wan2.1/README.md
- MLX Examples（视频模型清单）：https://github.com/ml-explore/mlx-examples
- ComfyUI（支持 Apple Silicon/Ascend/Cambricon/Iluvitar；视频模型清单）：https://github.com/comfyanonymous/ComfyUI
- ComfyUI 量化文档（QuantizedTensor/FP8/PTQ）：https://github.com/Comfy-Org/ComfyUI/blob/master/QUANTIZATION.md
- ComfyUI-GGUF（DiT 量化/GGUF）：https://github.com/city96/ComfyUI-GGUF
- SageAttention（SageAttention3 Blackwell FP4/RTX5090 560T/CogVideoX）：https://github.com/thu-ml/SageAttention
- LightX2V（4 步蒸馏/NVFP4/4090D 与 H100 benchmark/国产 NPU 适配）：https://github.com/ModelTC/LightX2V
- lightx2v Wan2.1 蒸馏模型：https://huggingface.co/lightx2v/Wan2.1-Distill-Models
- NVIDIA RTX 5090 产品页（32GB GDDR7/512-bit/3352 FP4 TOPS/575W/FP4）：https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
- NVIDIA Jetson 模块清单（T5000/T4000）：https://developer.nvidia.com/embedded/jetson-modules
- NVIDIA Jetson Thor 产品页（2070/1200 TFLOPS FP4、128/64GB、273 GB/s、40–130W）：https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/
- Apple Mac Studio 技术规格（M3 Ultra 819 GB/s / M4 Max 410–546 GB/s / 媒体引擎）：https://www.apple.com/mac-studio/specs/
- TeaCache 论文：https://arxiv.org/abs/2411.19108

> 未抓取到的来源：Rockchip RK3588 官方产品页两次 404（`/a/en/products/RK3588.html`、Radxa wiki），其规格以「Rockchip RK3588 数据手册公开规格」形式引用；`web_search`（Brave）未配置，故无搜索引擎发现，部分近期博客/第三方 benchmark 可能未覆盖。
