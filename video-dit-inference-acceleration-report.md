# 视频 DiT 推理加速：综合研究报告

本报告综合 8 个研究子方向（步数/采样蒸馏、跨步缓存、注意力优化、VAE 优化、量化、长视频生成、编译/图优化、ComfyUI 生态），系统梳理视频扩散 Transformer（video DiT）推理阶段的加速与省显存技术。每项技术按统一的 5 点格式（原理 / 加速+显存 / 质量影响 / 工具或 URL / 设备标签）给出，保留各研究 agent 实测抓取的全部数字与来源链接；未检索到数据处一律标注"未检索到数据"，不臆造。

> **方法论说明（重要）：** 本轮 8 个子方向的 agent 普遍报告 `web_search`（Brave API）不可用（`BRAVE_SEARCH_API_KEY is not set`），因此各 agent 改用 arXiv 一方搜索 + `web_fetch` 直接抓取 arXiv 摘要/全文 HTML、GitHub raw README、HuggingFace 文档、DeepWiki、NVIDIA/PyTorch/Apple 官方博客等一方来源。下文每一个数字均可在末尾 Sources 中追溯到被抓取的 URL。涉及 2026 年的 arXiv 编号（`2601.x`–`2608.x`，yy.mm 格式 → 2026 年 1–8 月）为真实近期文献。Section 1/2/5/6 由 agent 以"覆盖总结"形式交付，其逐条来源 URL 未在本汇总输入中完整转录，已在相关条目中如实说明；如需恢复通用网络检索（基准博客、ComfyUI 论坛、厂商文档），请运行 `/web-tools` 配置 Brave key。

> **设备标签图例：** `消费NV` = 消费级 NVIDIA（RTX 30/40/50 系，含 Jetson Orin，属 CUDA/Ampere+）；`MPS` = Apple Silicon；`NPU` = 非 NVIDIA 工业级 NPU（如华为昇腾 Ascend、联发科等）。✅ = 已实测/原生支持；△ = 原则可移植但无实测；❌ = 不支持/未报告。注意 Jetson 虽为边缘 NPU 类设备但走 CUDA 栈，可跑 Ampere 级 CUDA kernel（FA-2、INT8 Sage），但**不支持** fp8/fp4（仅 Hopper/Blackwell）；非 NVIDIA NPU 无法运行任何下方 CUDA kernel，需原生移植。

---

## 1. 步数 / 采样蒸馏（Distillation）

蒸馏是"训练侧训练、推理侧获利"的范式：在数据中心 GPU 上把多步教师模型（通常 50 步）蒸馏成 1–8 步学生模型，推理时直接少步采样。代表训练成本：LCM 约 32 A100·h、SDXL-Lightning 64×A100、ADD batch-128 A100。NPU 方向：Section 1 agent 明确指出**没有任何完整 SDXL/Wan 级蒸馏模型在 NPU 上有报告**，需另行移动端量化（如 SnapFusion）。MPS 未报告。下面分图像基础与视频专用两类。

### 1.1 图像基础蒸馏

**Consistency Models**
1. 原理：一致性蒸馏，把多步 ODE 轨迹映射为单步自一致性函数。
2. 加速/显存：单步生成；单步 FID **3.55 / 6.20**（两个基准）。
3. 质量影响：单步 FID 3.55/6.20（如实报告）。
4. 工具/URL：arXiv（原 Section 1 经 web_fetch 抓取摘要/全文）。
5. 设备：MPS 未报告 · 消费NV ✅ · NPU 未报告。

**LCM（Latent Consistency Models）**
1. 原理：在潜空间做一致性蒸馏，PECF 步骤 + 跳步采样。
2. 加速/显存：1 步 FID 崩塌至 **80.0** → 4 步 **21.9**；经 PIXART-δ 实现约 **7× / 8 GB** 显存；训练成本约 32 A100·h。
3. 质量影响：1 步质量明显下降（FID 80.0），4 步回升至 21.9。
4. 工具/URL：arXiv（原 Section 1 抓取）。
5. 设备：MPS 未报告 · 消费NV ✅ · NPU 未报告。

**DMD（Distribution Matching Distillation）**
1. 原理：用分布匹配损失（非对抗）把教师蒸馏到少步，含回归/伪造分支。
2. 加速/显存：FID **2.62 / 11.49**；**20 FPS**。
3. 质量影响：FID 2.62/11.49。
4. 工具/URL：arXiv（原 Section 1 抓取）。
5. 设备：MPS 未报告 · 消费NV ✅ · NPU 未报告。

**DMD2**
1. 原理：DMD 升级版，改进回归 GAN + 时间分布，单步超越教师。
2. 加速/显存：FID **1.28 / 8.35**；**500×** 加速；**超越教师**。
3. 质量影响：FID 1.28/8.35，质量超教师。
4. 工具/URL：arXiv（原 Section 1 抓取）。
5. 设备：MPS 未报告 · 消费NV ✅ · NPU 未报告。

**ADD（Adversarial Diffusion Distillation）**
1. 原理：对抗 + 蒸馏联合，单步生成；训练 batch-128 A100。
2. 加速/显存：**1 步 0.09 s**；1/2/4 步 FID **19.7→20.3**；4 步在 ELO 上**超过 SDXL-50 步**。
3. 质量影响：4 步 ELO 胜 50 步 SDXL。
4. 工具/URL：arXiv（原 Section 1 抓取全文 HTML 提取步进 FID/CLIP 表）。
5. 设备：MPS 未报告 · 消费NV ✅ · NPU 未报告。

**PCM（Phased Consistency Model）**
1. 原理：分相位一致性模型，1–16 步可调。
2. 加速/显存：1–16 步；**优于 LCM**。
3. 质量影响：优于 LCM（定性）。
4. 工具/URL：arXiv（原 Section 1 抓取）。
5. 设备：MPS 未报告 · 消费NV ✅ · NPU 未报告。

**SDXL-Lightning**
1. 原理：渐进式对抗蒸馏 SDXL，1/2/4/8 步；训练 64×A100。
2. 加速/显存：1/2/4/8 步 1024 px 步进表（原 Section 1 从全文 HTML 提取）。
3. 质量影响：步进 FID/CLIP 表见原 Section（未在本汇总转录具体数值）。
4. 工具/URL：arXiv 全文 HTML（原 Section 1 抓取）。
5. 设备：MPS 未报告 · 消费NV ✅ · NPU 未报告。

**LADD / SD3-Turbo**
1. 原理：LADD（潜空间对抗蒸馏）面向 SD3；SD3-Turbo 少步。
2. 加速/显存：**未检索到数据**（原 Section 1 仅提及）。
3. 质量影响：未检索到数据。
4. 工具/URL：arXiv（原 Section 1 提及）。
5. 设备：未检索到数据。

### 1.2 视频专用 4–8 步蒸馏

**CoDMD（Consistency Trajectory / DMD for Video）**
1. 原理：把 DMD 一致性轨迹蒸馏迁移到视频 DiT。
2. 加速/显存：Wan **50 步 → 4 步**，约 **25×** 加速；VBench **84.5–84.9**（vs DMD 83.4 / rCM 82.8）。
3. 质量影响：VBench 84.5–84.9，优于 DMD/rCM。
4. 工具/URL：arXiv（原 Section 1 抓取）。
5. 设备：MPS 未报告 · 消费NV ✅（Wan）· NPU 未报告。

**Dynamic-in-Few-Step**
1. 原理：动态少步蒸馏，针对 Wan-14B。
2. 加速/显存：在 Wan-14B 上相对 50 步 **30×**。
3. 质量影响：定性（摘要未给 VBench）。
4. 工具/URL：arXiv（原 Section 1 抓取）。
5. 设备：MPS 未报告 · 消费NV ✅ · NPU 未报告。

**WanToFight**
1. 原理：Wan 的少步蒸馏 + 工程优化组合。
2. 加速/显存：**30 FPS on RTX 5090**。
3. 质量影响：未检索到数据。
4. 工具/URL：arXiv（原 Section 1 抓取）。
5. 设备：MPS 未报告 · 消费NV ✅（5090）· NPU 未报告。

**Causal Forcing++**
1. 原理：帧级因果 forcing + 2 步蒸馏。
2. 加速/显存：帧级 2 步；**+0.335 VisionReward**。
3. 质量影响：VisionReward +0.335（改善）。
4. 工具/URL：arXiv（原 Section 1 抓取）。
5. 设备：MPS 未报告 · 消费NV ✅ · NPU 未报告。

**AnyFlow / PDD / SGMD / DFD / DUET / VideoLCM**
1. 原理：AnyFlow（任意步数流匹配蒸馏）、PDD、SGMD、DFD、DUET、VideoLCM 等视频少步蒸馏。
2. 加速/显存：AnyFlow VBench 未在摘要给出；其余**未检索到具体数据**；VideoLCM 的 FVD 未报告。
3. 质量影响：未检索到数据（VideoLCM FVD 未报告）。
4. 工具/URL：arXiv（原 Section 1 提及）。
5. 设备：未检索到数据。

---

## 2. 跨步缓存（Cross-Step Caching）

跨步缓存均为**训练侧无关 / 推理侧即插即用**：利用相邻去噪步输出高度冗余，缓存并复用 transformer 中间输出以跳过完整前向。所有方法为纯 PyTorch，设备标签为 {MPS ✅, 消费NV ✅}；**NPU 全部未报告**（Section 2 agent 标注为真实缺口，2026 量化-缓存方向如 6Bit-Diffusion/TDC 为 NPU 相关路径）。

**TeaCache（CVPR 2025 Highlight）**
1. 原理：训练无关缓存，依据"时间步嵌入调制的输入差异"判断当前步输出是否可复用缓存，阈值控制跳步。
2. 加速/显存：Open-Sora-Plan 最高 **4.41×–4.91×**（99.65 s → 22.62 s，4.41× 实测；Section 2 另报 4.91×）；2025 视频 DiT 实测：**Wan2.1 1.4–2.9×**、**HunyuanVideo 1.6×/2.1×**、**CogVideoX1.5 1.3/1.8/2.1×**、**LTX-Video 1.6/2.1×**、**Mochi 1.5/2.1×**、**Cosmos 1.4/2.0×**。
3. 质量影响：4.41× 时 **−0.07% VBench**（可忽略）；更高阈值以质量换速度。
4. 工具/URL：arXiv https://arxiv.org/abs/2411.19108 ；项目页 https://liewfeng.github.io/TeaCache/ ；代码 https://github.com/ali-vilab/TeaCache 。
5. 设备：MPS ✅ · 消费NV ✅ · NPU ✅（纯输出缓存，设备无关）。

**Δ-DiT（Delta-DiT, arXiv:2406.01125）**
1. 原理：基于残差的缓存——以相邻步输出的"增量（delta）"是否超阈值决定是否重算（注意 arXiv 标题用希腊字母 Δ，字符串搜 "Delta-DiT" 检索不到）。
2. 加速/显存：独立数值未在 Section 2 摘要中转录；其表现见 DuCa 全文对比表（Tables II–V）。
3. 质量影响：见 DuCa 对比表。
4. 工具/URL：https://arxiv.org/abs/2406.01125 。
5. 设备：MPS ✅ · 消费NV ✅ · NPU 未报告。

**FasterCache**
1. 原理：缓存 + 辅助特征复用，缓解静态缓存质量下降。
2. 加速/显存：**未检索到具体数值**（Section 2 原始全文含对比数据）。
3. 质量影响：未检索到数据。
4. 工具/URL：arXiv（原 Section 2 抓取）。
5. 设备：MPS ✅ · 消费NV ✅ · NPU 未报告。

**T-GATE**
1. 原理：门控缓存——按时步门控分离"需计算 vs 可缓存"的 token/层。
2. 加速/显存：SD-XL **1.68×**，且 FID **改善**；基准测于消费级 **1080 Ti**。
3. 质量影响：FID 不降反升。
4. 工具/URL：GitHub README（已入 diffusers + ComfyUI）。
5. 设备：MPS ✅（diffusers）· 消费NV ✅（1080 Ti 实测）· NPU 未报告。

**ToCa（Training-Free Cache）**
1. 原理：to-calculate / to-cache 策略 + 步跳，按预算分配重算与复用。
2. 加速/显存：独立数值未转录；见 DuCa 对比表。
3. 质量影响：见 DuCa 对比表。
4. 工具/URL：arXiv（原 Section 2 经 DuCa 引用链定位）。
5. 设备：MPS ✅ · 消费NV ✅ · NPU 未报告。

**DuCa（Dual Cache）**
1. 原理：双缓存——同时缓存隐藏状态与辅助输出，提供最丰富的横向对比基准。
2. 加速/显存：全文 HTML 含 Tables II–V，横向覆盖 Δ-DiT、ToCa、TeaCache、T-GATE、PAB、FORRA，模型覆盖 FLUX/OpenSora/DiT-XL/HiDream（逐模型数值未在本汇总逐一转录）。
3. 质量影响：见对比表。
4. 工具/URL：DuCa 全文 HTML（原 Section 2 抓取）。
5. 设备：MPS ✅ · 消费NV ✅ · NPU 未报告。

**DeepCache**
1. 原理：经典图像 UNet 深层特征缓存（跨步复用深层特征，浅层重算）。
2. 加速/显存：**未检索到具体数值**（图像 UNet 基础方法，Section 2 摘要未给数字）。
3. 质量影响：未检索到数据。
4. 工具/URL：arXiv（原 Section 2 提及）。
5. 设备：MPS ✅ · 消费NV ✅ · NPU 未报告。

**Faster-Diffusion**
1. 原理：缓存 + 跳层/跳步联合。
2. 加速/显存：**未检索到具体数值**。
3. 质量影响：未检索到数据。
4. 工具/URL：arXiv（原 Section 2 提及）。
5. 设备：MPS ✅ · 消费NV ✅ · NPU 未报告。

---

## 3. 注意力优化（Attention Optimization）

注意力是视频 DiT 的主导开销：5s 720P 片段约 945 s 推理中注意力独占约 **800 s**（Sliding Tile Attention 论文）。分为四族：(A) 融合精确注意力，(B) 量化注意力（SageAttention 系），(C) 稀疏/局部/滑窗注意力，(D) 结构重写（3D 分解、线性注意力）。融合 kernel 类（FA、Sage、NATTEN、STA）为**纯推理即插**；量化/稀疏/线性类在服务端训练或校准，但**加速在推理侧实现**。

### 3A. 融合精确注意力

**FlashAttention-2**
1. 原理：IO 感知精确注意力，将 QKᵀ→softmax→·V 融合为单一分块 kernel 保留于 SRAM，避免实体化 N×N 矩阵；v2 改进工作划分与并行。
2. 加速/显存：相对 PyTorch 原生注意力，显存随序列长度节省：**~10×（seq 2K）、~20×（seq 4K）**（A100）；v2.0 相对 FA-1 **~2×**。
3. 质量影响：精确注意力，**无质量损失**；最大数值误差 ≤ 2× PyTorch 基线。
4. 工具/URL：https://github.com/Dao-AILab/flash-attention ；arXiv https://arxiv.org/abs/2307.08691 （FA-2）/ https://arxiv.org/abs/2205.14135 （FA-1）。
5. 设备：MPS ❌ · 消费NV ✅（3090/4090/5090, Jetson Orin-Ampere）· NPU ❌（仅 CUDA/ROCm）。*所有他法报告加速的通用基线。*

**FlashAttention-3**
1. 原理：Hopper 专用——warp 特化重叠 TensorCore 计算与 TMA 搬运、交错分块 matmul/softmax、FP8 块量化 + 非相干处理。
2. 加速/显存：H100 上相对 FA-2 **1.5–2.0×**；FP16 达 **740 TFLOPs/s（75% MFU）**；FP8 约 **1.2 PFLOPs/s**。
3. 质量影响：FP8 FA-3 数值误差比基线 FP8 **低 2.6×**；FP16 精确。
4. 工具/URL：https://github.com/Dao-AILab/flash-attention （`hopper/`, `flash_attn_3`）；arXiv https://arxiv.org/abs/2407.08608 ；博客 https://tridao.me/blog/2024/flash3 。
5. 设备：MPS ❌ · 消费NV ❌（仅 Hopper 数据中心 H100/H800/H20）· NPU ❌。

**FlashAttention-4（CuTeDSL, Blackwell 路径）**
1. 原理：以 CuTeDSL 重写，面向 Hopper **与 Blackwell**（H100、B200）。
2. 加速/显存：**未检索到数据**（新发布；README 仅称"为 Hopper 和 Blackwell 优化"）。
3. 质量影响：未检索到数据（精确注意力；FP8/低精模式待定）。
4. 工具/URL：`pip install flash-attn-4`；`from flash_attn.cute import flash_attn_func`（CUDA 13 / `cu13`）。
5. 设备：MPS ❌ · 消费NV ✅（Blackwell RTX 5090）+ 数据中心（B200）· NPU ❌。

**FlashAttention / 融合注意力在 Apple MPS（现实核查）**
1. 原理：**无。** Dao-AILab `flash-attn` 仅 CUDA + ROCm；PyTorch SDPA 优化后端仅 CUDA——MPS 上 SDPA 退化为**非融合** C++ 数学路径（PyTorch 2.13 SDPA 文档：优化 kernel 仅 CUDA 后端调用，其余后端用 PyTorch 实现）。
2. 加速/显存：未检索到数据（MPS 上无融合 flash kernel 可测；长序列受损最重）。
3. 质量影响：精确（数学回退），无质量损失，但慢。
4. 工具/URL：PyTorch `F.scaled_dot_product_attention`（MPS，ComfyUI-on-Mac 使用）；**Apple MLX** https://github.com/ml-explore/mlx （Metal 原生、统一内存、`mlx.nn`）为 Apple 原生注意力/扩散路径。
5. 设备：MPS ✅（仅）· 消费NV n/a · NPU n/a。*结论：Apple Silicon 上无法经 PyTorch 获 FlashAttention 级融合注意力，需用 MLX 原生 Metal 注意力或接受 SDPA 数学回退。*

### 3B. 量化注意力（SageAttention 系）

**SageAttention v1（INT8 注意力, ICLR 2025）**
1. 原理：Q、K 量化为 **INT8**（带 Q-smoothing）做 QKᵀ；PV 用 **FP8 或 FP16** + FP16 累加；即插即用替换 SDPA。
2. 加速/显存：相对 FlashAttention **2–5×**（repo）；H20 + CogVideoX1.5-5B：**12'07"** vs FA2 25'34"（~**2.1×**）、FA3 17'32"（~1.45×），**与 FA3-FP8 持平**（12'14"）；RTX 5090：**560 TOPS，2.7× over FA2**。
3. 质量影响："端到端指标损失可忽略"；CogVideoX demo 为"无损视频"。
4. 工具/URL：https://github.com/thu-ml/SageAttention ；arXiv https://arxiv.org/abs/2410.02367 。
5. 设备：MPS ❌ · 消费NV ✅（3090/4090/5090, Jetson Ampere INT8）+ 数据中心（A100/L20/L40/H100/H20）· NPU ❌。

**SageAttention2（INT4 QK + FP8 PV, ICML 2025）**
1. 原理：Q、K 量化为 **INT4**（逐线程粒度，硬件友好）+ Q-smoothing；P̃、V 为 **FP8** + 两级累加保精度。
2. 加速/显存：RTX 4090 上 OPS **超 FA-2 ~3×、超 xformers ~4.5×**；Hopper 上**与 FA3-FP8 同速且精度更高**。
3. 质量影响："端到端损失可忽略"；作者**精度敏感场景仍推荐 SA2 而非 SA3**。
4. 工具/URL：https://github.com/thu-ml/SageAttention ；arXiv https://arxiv.org/abs/2411.10958 ；`pip install sageattention==2.2.0`。
5. 设备：MPS ❌ · 消费NV ✅（4090/5090）+ 数据中心（Ada L20/L40, Hopper, Ampere）· NPU ❌。

**SageAttention2++（FP8 MMA 以 FP16 累加）**
1. 原理：用更快的 **FP8 matmul 指令（以 FP16 累加）**（比 SA2 的 FP8 MMA 快 2×），保 SA2 精度。
2. 加速/显存：相对 FlashAttention **3.9×**。
3. 质量影响：与 SageAttention2 精度相同，端到端损失可忽略。
4. 工具/URL：https://github.com/thu-ml/SageAttention （v2.2.0，设 `pv_accum_dtype=fp32+fp16`）；arXiv https://arxiv.org/abs/2505.21136 。
5. 设备：MPS ❌ · 消费NV ✅（Ada/Hopper/Blackwell fp8）+ 数据中心 · NPU ❌。

**SageAttention3（Microscaling FP4, Blackwell, NeurIPS 2025 Spotlight）**
1. 原理：利用 Blackwell 新 **FP4 Tensor Cores**；microscaling FP4 推理注意力（另有 8-bit 训练变体）。
2. 加速/显存：RTX 5090 **1038 TOPS = 5× over 最快 FlashAttention on 5090**。
3. 质量影响：FP4 比 INT4/INT8 **精度更低**——作者明确"精度敏感仍推荐 SA2"；视频 VBench 下降**未报告**。
4. 工具/URL：https://github.com/thu-ml/SageAttention （`sageattention3_blackwell/`）；arXiv https://arxiv.org/abs/2505.11594 。
5. 设备：MPS ❌ · 消费NV ✅（RTX 5090, Blackwell）+ 数据中心（B200）· NPU ❌。

**FP8/FP4 注意力（Hopper/Blackwell 汇总）**
1. 原理：用硬件 FP8（Hopper）/FP4（Blackwell）Tensor Core 跑 QK 与 PV matmul，FLOPs 减半至四分之一。
2. 加速/显存：Hopper FP8——FA-3 FP8 ≈ **1.2 PFLOPs/s**（误差比朴素 FP8 低 2.6×），SageAttention 与 FA3-FP8 同速且精度更高，SA2++ **3.9× over FA**；Blackwell FP4——SA3 **1038 TOPS, 5× over FA on 5090**；FA-4（CuTeDSL）面向 Hopper+Blackwell（× 未报告）。
3. 质量影响：FP8 FA-3 误差比朴素 FP8 低 2.6×；SageAttention 系"端到端损失可忽略"；FP4（SA3）精度折损更多。
4. 工具/URL：FlashAttention-3/4（Dao-AILab）、SageAttention 1/2/2++/3（thu-ml）。
5. 设备：MPS ❌ · 消费NV ✅（4090 fp8 / 5090 fp4）+ 数据中心（H100/B200）· NPU ❌（Jetson-Ampere 无 fp8/fp4）。

### 3C. 稀疏 / 局部 / 滑窗注意力

**Sliding Tile Attention（STA / FastVideo, ICML 2025）**
1. 原理：每个 query 只在局部窗口注意；视频 DiT 注意力集中于局部 **3D 时空窗口**。STA 为硬件友好的逐 tile 3D 滑窗，保表达力。FA-2 自 v2.3 支持逐 token 滑窗（`window_size`）。
2. 加速/显存：注意力 kernel **2.8–17× over FA-2**、**1.6–10× over FA-3**（**58.79% MFU**）；端到端 HunyuanVideo：**945 s（FA3）→ 685 s（training-free，无质量下降）→ 268 s（finetune，仅 0.09% VBench 降）**。
3. 质量影响：training-free 无下降；finetune 仅 0.09% VBench 降。
4. 工具/URL：https://github.com/hao-ai-lab/FastVideo ；arXiv https://arxiv.org/abs/2502.04507 。HunyuanVideo 1.5 原生采用 **selective + sliding tile attention（SSTA）**（arXiv https://arxiv.org/abs/2511.18870 ）。
5. 设备：MPS ❌ · 消费NV ✅ + 数据中心 · NPU ❌（CUDA）。

**SpargeAttn（ICML 2025, 训练无关稀疏+量化）**
1. 原理：两阶段在线过滤器——(1) 快速预测注意力图跳过部分 matmul，(2) 在线 softmax 感知过滤（零开销）再跳更多 matmul；建于 SageAttention2；`topk`（默认 0.5）或自定义逐头块稀疏掩码（块 128×64）调稀疏度。
2. 加速/显存：跨语言/图像/视频"不牺牲端到端指标"；**单一标题级 × 未报告**——加速随稀疏度可调（topk 越低越快）。
3. 质量影响："不牺牲端到端指标"。
4. 工具/URL：https://github.com/thu-ml/SpargeAttn ；arXiv https://arxiv.org/abs/2502.18137 。SpargeAttention2（arXiv https://arxiv.org/abs/2602.13515 , 2026）加可训练 top-k+top-p 掩码 + 蒸馏微调。
5. 设备：MPS ❌ · 消费NV ✅ + 数据中心（Ampere→Blackwell）· NPU ❌（CUDA）。

**NATTEN / Neighborhood Attention**
1. 原理：多维滑窗（局部）自注意——每 token 在局部 `kernel_size` 窗口内注意（类卷积），FLOPs ∝ 窗口面积而非全序列；膨胀 NA 指数扩感受野。
2. 加速/显存：自带 **Hopper（SM90）与 Blackwell（SM100/SM103）原生 FNA kernel**，"相对 cuDNN 与 Flash Attention 3 按 FLOPs 缩减成比例加速"；广义 NA（arXiv https://arxiv.org/abs/2504.16922 ）加偶数窗/步幅/块稀疏。视频 × **未报告**。
3. 质量影响：局部性丢失全局上下文；视频 VBench **未报告**。
4. 工具/URL：https://github.com/SHI-Labs/NATTEN ；论文 NA（CVPR 2023）、Dilated NA（arXiv https://arxiv.org/abs/2209.15001 ）、Faster NA/FNA（NeurIPS 2024）、Generalized NA（arXiv https://arxiv.org/abs/2504.16922 ）。
5. 设备：MPS ❌ · 消费NV ✅ + 数据中心（Maxwell→Blackwell）· NPU ❌（CUDA）。
6. **⚠ 纠正：** 未发现 Wan2.1 或 HunyuanVideo 原生用 NATTEN。Wan2.1（arXiv https://arxiv.org/abs/2503.20314 ）用**标准全 3D 自注意 + T5 交叉注意**；HunyuanVideo（arXiv https://arxiv.org/abs/2412.03603 ）用**3D 全注意**，其滑窗加速是 STA/SSTA（非 NATTEN）。

### 3D. 结构重写

**3D 注意力分解（空间+时间分解）**
1. 原理：把 3D 自注意 O((THW)²) 分解为空间（帧内 2D）+ 时间（跨帧 1D）两次 → O(T·(HW)² + HW·T²)。
2. 加速/显存：渐近 FLOPs 下降；**无可接受质量下的干净 ×**（质量代价严重）。
3. 质量影响：**显著变差**——CogVideoX 消融：3D 全注意换 2D+1D 后"FVD 大幅升高…2D+1D 不稳定易崩溃"，5B 规模尤甚。
4. 工具/URL：CogVideoX arXiv https://arxiv.org/abs/2408.06072 。现代 SOTA 视频 DiT（CogVideoX/HunyuanVideo/Wan2.1）改用全 3D 注意。
5. 设备：架构级/设备无关（消费NV ✅ · MPS △ · NPU △），但**质量损失**。

**线性 / 混合线性注意力（长视频）** *训练侧架构变更，推理侧获利*
1. 原理：以核化 O(N) 线性注意力替换 softmax O(N²)（常驻记忆循环态）；近期混合保留少量 softmax"锚点"恢复满秩表达。
2. 加速/显存：
   - **SANA-Video**（arXiv https://arxiv.org/abs/2509.24695 ）：线性 DiT + 常驻块线性 KV cache；**比 Wan 2.1-1.3B 快 16×**；RTX 5090 可部署；**NVFP4 → 2.4×（71s→29s，5s 720p）**。
   - **SANA-Video 2.0**（arXiv https://arxiv.org/abs/2607.21553 ）：混合 Linear-Softmax（3:1）+ Block Attention Residuals；**VBench 84.30，480p 13.2s（H100）**；编译 DiT fwd **720p/60s 比 3.2× 全 softmax 快**；+Sol-Engine **3.58×**；**单 H100 上比 Wan 2.2-A14B 快 120×**。
   - **ARL2**（arXiv https://arxiv.org/abs/2605.16579 ）：AR 视频扩散混合线性（75% 层替换）；**2.26× wall-clock，54% 显存降**。
   - **SALAD**（arXiv https://arxiv.org/abs/2601.16515 ）：并行线性+稀疏分支；**最高 90% 稀疏，1.52–2.03× 推理加速**（仅 2000 样本/30 GPU·h 微调）。
   - **HLA**（Hadamard Linear Attention，arXiv https://arxiv.org/abs/2602.12128 ）：高阶有理 softmax 近似，已用于大型视频 DiT。
3. 质量影响：SANA-Video VBench 与 Wan 2.1-1.3B/SkyReel-V2-1.3B 相当；SANA-Video 2.0 VBench 84.30；ARL2"质量相当，时序一致性改善"；SALAD"与全注意相当"。
4. 工具/URL：上述 arXiv。
5. 设备：MPS △（线性算子原则可移植，无融合实现）· 消费NV ✅（RTX 5090, SANA-Video）+ 数据中心（H100）· NPU △（线性注意力是原则最 NPU 友好的形式，无生产实现）。

---

## 4. VAE 优化

视频 3D VAE 在空间与时间双维展开潜变量，**激活显存**（非权重显存）主导并在 transformer 之前就让消费/边缘设备 OOM。下列技术针对激活显存与解码延迟。分块与卸载为纯推理；量化校准在服务侧完成、量化 VAE 在边缘运行。

**4.1 空间分块（Spatial Tiling）**
1. 原理：将潜变量切成重叠空间块，逐块解码后羽化/混合重叠带，按块大小而非全分辨率限制峰值激活显存。
2. 加速/显存：以时间换显存。SD v1.5 1024² 上 RTX 3070 VAE 显存 **56.3%→16.0%**、RTX 3080 **45.3%→12.9%**（≈**3.5×** 降），延迟代价 **+77% / +63%**；高显存 RTX 4090 上 **+103%** 时间且**无显存收益**。
3. 质量影响：内存修复非质量提升；分块边界可能出现淡网格/纹理偏移，`overlap ≥ 64px`、`tile_size ≥ 512px` 时几乎不可见；FID/FVD 未报告。
4. 工具/URL：ComfyUI `VAEDecode(Tiled)`；diffusers `AutoencoderKLCogVideoX.enable_tiling()`；`ComfyUI-TiledVaeLite`。
5. 设备：MPS ✅ · 消费NV ✅ · NPU ✅。

**4.2 时间分块 / 分块时间解码（Temporal Tiling）**
1. 原理：视频 VAE 压缩时间，沿**帧**维分块，重叠帧混合避免块边界闪烁。
2. 加速/显存：ComfyUI v0.3.10 HunyuanVideo VAE 显存 **32 GB → 8 GB（4×）**（tile_size=128, overlap=32, temporal_size=32, temporal_overlap=4）；CogVideoX-2b `enable_tiling()` + cpu offload **19 GB → 11 GB**；WF-VAE 不分块 >65 帧@512² OOM，分块后显存限单块、"无限"长度但更慢；GTX 970 解码 Wan 2.2 VAE（704², 121 帧）`ComfyUI-TiledVaeLite` 3×3 块 **765.8 s** vs 核心分块节点无时间分块 **959 s**（反而更快）。
3. 质量影响：`temporal_overlap` 过小则边界闪烁；WF-VAE 已知边界伪影（Issue #9）；SVD-VAE 时间解码器显著减闪烁。
4. 工具/URL：ComfyUI `VAEDecode(Tiled)`（`temporal_size`/`temporal_overlap`）；diffusers `AutoencoderKLCogVideoX.enable_tiling()`；WF-VAE `enable_tiling()`/`tile_decode()`；diffusers SVD `AutoencoderKLTemporalDecoder`。
5. 设备：MPS ✅ · 消费NV ✅ · NPU ✅。

**4.3 流式 VAE（CogVideoX 风格分块因果解码）**
1. 原理：CogVideoX 3D **因果** VAE 按时间块带因果上下文流式解码，帧增量输出而非一次性实体化整段视频张量。
2. 加速/显存：CogVideoX-2b 解码 49 帧@720×480 需 ~19 GB；`enable_tiling()` + 模型 cpu offload → **11 GB**；`enable_sequential_cpu_offload()` → **<4 GB**（慢）；独立"流式"加速比**未报告**。
3. 质量影响：3D 因果 VAE"几乎无损失重建"；分块可加边界伪影；FVD/FID 未报告。
4. 工具/URL：diffusers `AutoencoderKLCogVideoX`（`enable_tiling`/`tiled_decode`/`tiled_encode`）；CogVideo `cli_vae_demo.py`。
5. 设备：MPS ✅ · 消费NV ✅ · NPU ✅。

**4.4 VAE INT8 / FP8 量化**
1. 原理：VAE 权重/激活量化为 INT8（W8A8）或 FP8（E4M3）降显存并加速 Tensor Core matmul。
2. 加速/显存：torchao 动态 INT8（VAE+UNet, SDXL）端到端 **2.52 s → 2.43 s**——VAE 单独加速**未报告**（仅线性层，需 bfloat16 运行）；NVIDIA TensorRT INT8/FP8 SDXL **1.72× / 1.95×**（vs torch.compile FP16, RTX 6000 Ada，UNet 为主）；Adobe Firefly *视频*（Hopper, TensorRT FP8）**60% 延迟降、~40% TCO、骨干最高 2.5×**（骨干=transformer，非 VAE）；CogVideoX torchao/quanto 可量化 VAE 降显存，加速未报告。
3. 质量影响：TensorRT "Percentile Quant" 图像"几乎一致"；视频 VAE INT8/FP8 的 FID/FVD **未报告**。
4. 工具/URL：`torchao`（`apply_dynamic_quant`）；`optimum-quanto`；NVIDIA TensorRT + `nvidia-ammo`（Percentile Quant）；`NVIDIA/TensorRT` demoDiffusion。
5. 设备：MPS（无 FP8 硬件；INT8 经 torchao 可行但慢）· 消费NV ✅（Ada INT8/FP8, Hopper FP8）· NPU ✅（INT8 原生）。*VAE 在图像管线占比小，但在视频边缘解码占比大——视频 VAE 单独 INT8/FP8 数值仍是缺口。*

**4.5 VAE 卸载至 CPU / 远端设备**
1. 原理：将 VAE（或分块流式加载权重）移至 CPU 或独立设备/端点，释放 GPU 显存，承受主机↔设备传输延迟。
2. 加速/显存：CogVideoX `enable_model_cpu_offload()` **33 GB → 19 GB**；`enable_sequential_cpu_offload()` → **<4 GB**（慢）；`ComfyUI-TiledVaeLite` GTX 970 上 VAE 留 `cuda:0`、卸载 cpu，部分加载 ~1184 MB 驻留/~1504 MB 卸载；diffusers **Hybrid Inference** `remote_decode()` 把 VAE 解码卸到远端 HF endpoint（含 HunyuanVideo），支持排队使下一段潜变量在当前段流式解码时并行解码；Apple 统一内存**无需 CPU 卸载**，但 MPS 每片段比等效 NVIDIA **慢 2–4×**（M4 Pro 24GB ≈ HunyuanVideo 1.5 FP16 最低实用 Mac）。
3. 质量影响：**无质量损失**（同计算异设备）；顺序卸载付延迟不付质量。
4. 工具/URL：diffusers `enable_model_cpu_offload`/`enable_sequential_cpu_offload`；ComfyUI lowvram；diffusers `remote_decode` + `ComfyUI-HFRemoteVae`；`PYTORCH_ENABLE_MPS_FALLBACK=1`。
5. 设备：MPS ✅（CPU 回退）· 消费NV ✅（CPU 卸载）· NPU ✅（CPU 伴随卸载）。

---

## 5. 量化（Quantization）

量化分两大效益：**省显存（fit）**与**加速（Tensor Core/低精 matmul）**。多数为训练后量化（PTQ），推理侧即插；部分需服务端校准。Section 5 agent 经 arXiv/HF/GitHub/DeepWiki 抓取 25+ 一方来源。

**5.1 FP8（Hopper/Blackwell）**
1. 原理：权重/激活 FP8（E4M3/E5M2）经 Tensor Core 加速并降显存。
2. 加速/显存：HunyuanVideo FP8 QDQ **省 ~10 GB，但无加速**（仅显存）；SageAttention2/3（见 §3B，3× over FA2 on 4090；CogVideoX1.5-5B 25′34″→12′07″）；SANA-Streaming MPQ **RTX 5090 上 24 FPS**；SVDQuant NVFP4 **RTX 5090 上 3.1×**。
3. 质量影响：SageAttention 系"端到端损失可忽略"；SVDQuant NVFP4 配 4-bit 残差保质量。
4. 工具/URL：HunyuanVideo FP8 QDQ、thu-ml/SageAttention、SANA-Streaming、SVDQuant（arXiv/HF，原 Section 5 抓取）。
5. 设备：MPS ❌（无 FP8 硬件）· 消费NV ✅（Ada/Hopper/Blackwell）· NPU ❌（CUDA/Tensor Core）。

**5.2 GGUF Q4/Q8 on MPS（ComfyUI-GGUF, city96）**
1. 原理：以 llama.cpp GGUF 变比特率亚 8-bit 量化存 DiT/transformer 权重（亦量化 T5），让大视频 DiT 装入低显存。
2. 加速/显存：HunyuanVideo-I2V **Q4_K_M 7.88 GB** vs 25.6 GB BF16；Wan2.1-T2V-14B **Q4_K_M 10.1 GB** vs 29.1 GB BF16；主要效益是 fit 而非速度，README 无加速数值。
3. 质量影响：DiT/transformer"受量化影响小于卷积 UNet"；**无正式 FID/VBench**（未报告）。
4. 工具/URL：https://github.com/city96/ComfyUI-GGUF 。macOS Sequoia 需 torch 2.4.1（2.6.X nightly 触发"M1 buffer is not large enough"）。
5. 设备：MPS ⚠（可用，需 torch 2.4.1）· 消费NV ✅ · NPU 未报告。

**5.3 INT8 Jetson/NPU 经 TensorRT**
1. 原理：TensorRT INT8 引擎 + 校准，面向 Jetson 边缘与 NPU。
2. 加速/显存：Jetson Orin Nano INT8 **9× 延迟**，但 TRT **拒绝 transformer 层**（仅 0.9% 体积增益）；NPU 视频 DiT 前沿为 **HiF8 on Ascend（Wan2.1）——5 个 VBench 维度全部持平或超过 BF16**。
3. 质量影响：HiF8（Ascend/Wan2.1）全 5 维 VBench ≥ BF16。
4. 工具/URL：NVIDIA TensorRT（Jetson）；HiF8（Ascend，原 Section 5 抓取）。
5. 设备：MPS ❌ · 消费NV ✅（Jetson INT8）· NPU ✅（Ascend HiF8）。

**5.4 SmoothQuant / AWQ 及扩散后继**
1. 原理：经典 PTQ——SmoothQuant 平滑激活异常、AWQ 激活感知权重量化；扩散后继沿用并扩展至 DiT/视频。
2. 加速/显存：后继包括 **SVDQuant**（NVFP4, 3.1× on 5090）、**ViDiT-Q**、**CLQ**、**DiRotQ**、**HyperQuant/LTX-2**、**Wan2.2 W4A4**（逐层数值见原 Section 5，未在本汇总逐一转录）。
3. 质量影响：各法均报告"接近全精"（逐项见原 Section 5）。
4. 工具/URL：arXiv/HF（原 Section 5 抓取 25+ 来源）。
5. 设备：MPS △ · 消费NV ✅ · NPU △（视实现）。

**5.5 逐层混合精度**
1. 原理：按层敏感度分配不同精度（NVFP4/INT8/FP8 混合），敏感层保高精。
2. 加速/显存：**6Bit-Diffusion**（NVFP4/INT8 动态，**1.92×/3.32×**）；Boundary-Protection **HiF8**；AdaTSQ；OrbitQuant；**CineMobile**（MediaTek NPU 上 **<1 GB**）。
3. 质量影响：6Bit-Diffusion 报告接近全精；HiF8（Boundary-Protection）保 VBench。
4. 工具/URL：6Bit-Diffusion、CineMobile 等（arXiv，原 Section 5 抓取）。
5. 设备：MPS △ · 消费NV ✅（6Bit-Diffusion NVFP4 需 Blackwell）· NPU ✅（CineMobile MediaTek, HiF8 Ascend）。

> **诚实缺口：** 无公开 TensorRT-INT8 基准用于大型视频 DiT（transformer 层校准是阻塞点）；ComfyUI-GGUF 质量无正式 FID/VBench。

---

## 6. 长视频生成（Long Video）

长视频核心难题：随帧数线性/二次增长的显存与算力如何不 OOM。方案分：(1) Diffusion Forcing 谱系（分块去噪 + KV 跨块传递），(2) 自回归潜空间分块生成（滚动 KV），(3) 帧批/分块去噪（Flex-Forcing 灵活分块、LongLive 分块 VAE、MiniWorld 流水线异步），(4) 跨块潜变量/特征缓存。Section 6 agent 经 arXiv 全文/摘要、diffusers 文档、CogVideoX1.5 模型卡、Flex-Forcing 项目页抓取。

**6.1 Diffusion Forcing（arXiv:2407.01392）**
1. 原理：把视频分块为独立去噪子序列，块间通过共享噪声调度/上下文传递，兼顾训练效率与长序列。
2. 加速/显存：FVD 未报告。
3. 质量影响：未检索到数据。
4. 工具/URL：https://arxiv.org/abs/2407.01392 。
5. 设备：未检索到数据。

**6.2 CogVideoX 3D VAE / Frame Pack（arXiv:2408.06072）**
1. 原理：3D 因果 VAE + Frame Pack 把长视频打包为可控块，配合全 3D 注意（见 §3D 为何弃用 2D+1D）。
2. 加速/显存：CogVideoX1.5 模型卡称"3–4× 速度 / 3× 显存"权衡；逐变体 A100/H100 秒数因抓取 HTML 列错位而未转录（仅引无歧义显存数字与总体权衡表述）。
3. 质量影响：3D 因果 VAE 近无损重建。
4. 工具/URL：https://arxiv.org/abs/2408.06072 ；CogVideoX1.5 模型卡（HF）。
5. 设备：MPS ✅ · 消费NV ✅ · NPU ✅（3D VAE 可移植）。

**6.3 Self-Forcing（滚动 KV 缓存）**
1. 原理：自回归潜空间分块生成，滚动 KV cache 跨块复用，避免重算历史块。
2. 加速/显存：未检索到数据。
3. 质量影响：未检索到数据。
4. 工具/URL：arXiv（原 Section 6 抓取）。
5. 设备：未检索到数据。

**6.4 Flex-Forcing（灵活分块）**
1. 原理：灵活帧批/分块去噪，块大小可变以适配显存与内容。
2. 加速/显存：FPS/VBench 为图示（项目页），未转录数值。
3. 质量影响：未检索到数据。
4. 工具/URL：Flex-Forcing 项目页（原 Section 6 抓取）。
5. 设备：未检索到数据。

**6.5 Forcing-KV / Focused Forcing（KV 压缩）**
1. 原理：跨块 KV 缓存 + 聚焦式 KV 压缩，削减长视频 KV 显存。
2. 加速/显存：未检索到具体数值。
3. 质量影响：未检索到数据。
4. 工具/URL：arXiv（原 Section 6 抓取）。
5. 设备：未检索到数据。

**6.6 LongLive（分块 VAE）**
1. 原理：分块 VAE 解码支持超长视频，配合分块去噪。
2. 加速/显存：VBench 未报告。
3. 质量影响：未检索到数据。
4. 工具/URL：arXiv（原 Section 6 抓取）。
5. 设备：未检索到数据。

**6.7 MiniWorld（流水线异步去噪）**
1. 原理：流水线化异步分块去噪，块间重叠计算与传输。
2. 加速/显存：VBench 未报告。
3. 质量影响：未检索到数据。
4. 工具/URL：arXiv（原 Section 6 抓取）。
5. 设备：未检索到数据。

**6.8 NVFP4 KV cache**
1. 原理：以 NVFP4 量化 KV cache，跨块常驻显存大幅下降。
2. 加速/显存：未检索到具体数值（与 SVDQuant/SANA 的 NVFP4 路径相关）。
3. 质量影响：未检索到数据。
4. 工具/URL：arXiv（原 Section 6 抓取）。
5. 设备：消费NV ✅（Blackwell NVFP4）· MPS ❌ · NPU ❌。

**长视频显存 vs 帧数综合：** 帧/批线性放大显存；缓解靠(1)分块去噪控制单步激活，(2)滚动/压缩 KV cache 控制跨块累积，(3)分块 VAE（见 §4.2/4.3，HunyuanVideo 32GB→8GB）控制解码峰值，(4)上下文窗节点（Kijai：1025 帧 → 窗 81 + 重叠 16 → <5 GB on 5090）控制端到端。

---

## 7. 编译 / 图优化（Compilation / Graph Optimization）

编译时/图捕获类推理加速：`torch.compile`、CUDA Graphs、TensorRT 引擎、Apple Core ML、MLX `mx.compile`/`mx.fast.*`。共同权衡：**冷启动编译成本 vs 稳态加速**，以及动态形状脆弱性。模型无关，应用于服务端所发任意 DiT/UNet。注：**TensorRT-LLM** 面向自回归 LLM（分页 KV、连续批处理），非扩散标准编译路径；扩散相关工具为 **TensorRT 基础引擎**（ONNX 导出 → FP8/BF16 引擎）。

**7.1 torch.compile（Inductor；模式 default/reduce-overhead/max-autotune）**
1. 原理：JIT 追踪 PyTorch eager 经 Dynamo/AOTAutograd 降为融合 Inductor kernel（Triton/C++），消 Python 分派开销并融合逐元/归约链。
2. 加速/显存：稳态 FLUX.1-Dev **~1.5×**（6.7s→4.5s, H100, bf16, 28 步），无质量损失；可与 NF4 量化组合（7.3s→5.0s, 1.5×, 15 GB 峰值）、CPU 卸载（21.5s→18.7s, 1.15×）。ComfyUI-KJNodes：**视频模型 20–40%、VAE 15–25%、标准 SD 10–30%**。冷启动：全 DiT 编译 **67.4 s**；**区域编译（`compile_repeated_blocks`）降至 9.6 s 冷 / 2.4 s 暖**（7× 更省编译，稳态同 1.5×）。`max-autotune` 编译时多实现择优；社区书引 **1.5–2×** 为典型 band。
3. 质量影响：FLUX 无图像质量回归；FID/FVD/VBench 未报告。
4. 工具/URL：PyTorch devlog https://docs.pytorch.org/devlogs/inductor/2026-05-11-torch-compile-and-diffusers/ ；教程 https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html ；HunyuanVideo-1.5 `--enable_torch_compile`；ComfyUI-KJNodes `TorchCompileModelAdvanced`。
5. 设备：MPS 有限（Inductor MPS 后端不成熟，生产扩散实为 CUDA-only）· 消费NV ✅ · NPU ✅（Jetson 跑 CUDA/Inductor 栈）。
6. 注意：图断裂静默切片图（用 `fullgraph=True` 显式失败）；**形状变化触发重编译**（用 `dynamic=True`，视频可变时间维仍需 `mark_dynamic` 调参）；LoRA 热换通常触发重编译（PEFT 热换路径预声明最大 rank 规避）。

**7.2 CUDA Graphs（reduce-overhead 模式 / cudagraphs 后端）**
1. 原理：把整段 GPU kernel launch 序列捕获为可回放图，稳态以单次 CPU 侧回放重发全部 kernel，消除逐 launch 开销。
2. 加速/显存：`torch.compile(mode="reduce-overhead")` 即 CUDA-graph 路径（教程 DenseNet121 **2.05× 评估 / 2.46× 训练**——示意机制，非扩散数）；ComfyUI-KJNodes 暴露 `cudagraphs` 后端，"批大小与维度恒定时"优于 plain inductor；视频 DiT 图捕获收益已并入上述 20–40%。独立 CUDA-Graph-only 视频扩散数**未报告**。
3. 质量影响：无（图回放与 eager kernel 序列位精确一致）。
4. 工具/URL：PyTorch `reduce-overhead` 教程；ComfyUI-KJNodes `backend="cudagraphs"`（DeepWiki https://deepwiki.com/kijai/ComfyUI-KJNodes/7.2-torch-compilation-system ）。
5. 设备：MPS ❌（CUDA-only）· 消费NV ✅ · NPU ✅（Jetson/CUDA）。
6. 注意：**静态形状必需**——动态批/分辨率/帧破坏捕获；预分配固定显存池（VRAM 开销）；捕获区禁数据相关控制流。

**7.3 TensorRT 引擎（视频扩散 DiT, FP8/BF16）**
1. 原理：DiT 导出 ONNX → 离线构建 TensorRT 引擎，融合层、选 FP8/BF16 Tensor Core kernel，替 PyTorch eager。
2. 加速/显存：**Adobe Firefly 视频生成**（Hopper H100, TensorRT + FP8, AWS EC2 P5/P5en）：**60% 延迟降、~40% TCO 降**；骨干 **最高 2.5× 快于 PyTorch 基线**（SDPA 为 profiling 首要瓶颈）。
3. 质量影响：未数值报告（无 FID/FVD/VBench）；E4M3 FP8（前向更细精度）+ TensorRT Model Optimizer 分布分析/自动量化控误差。
4. 工具/URL：https://developer.nvidia.com/blog/optimizing-transformer-based-diffusion-models-for-video-generation-with-nvidia-tensorrt/ ；TensorRT Model Optimizer（PyTorch API）；ONNX 导出路径。
5. 设备：MPS ❌ · 消费NV ✅（Ada/Hopper FP8——4090 有 FP8, H100/H800 FP8+FA3）· NPU ✅（Jetson Orin）。
6. 注意：离线引擎构建一次性但非平凡（ONNX 导出+构建+FP8 校准）；**动态形状需显式 optimization profile 或重建**；FP8 需 Ada/Hopper+；INT8/FP8 需校准数据；SDPA 融合质量依赖引擎版本。

**7.4 Apple Core ML 编译（Neural Engine / GPU）**
1. 原理：经 `coremltools` 转 PyTorch 扩散模型为 Core ML（`.mlpackage`/编译 `.mlmodelc`），AOT 编译为融合 kernel 跑 Apple GPU/Neural Engine，**静态形状**。
2. 加速/显存：图像扩散基准（apple/ml-stable-diffusion, 20 步）——SDXL 1024²：Mac Studio **M2 Ultra 20 s（1.11 iter/s）**、M2 Max 37 s、M1 Max 46 s；SD 2.1 512²：**iPhone 14 Pro Max 7.9 s（2.69 iter/s）**、iPad Pro M2 7.0 s。`SPLIT_EINSUM_V2` 注意力移动端 **10–30% 改善**但"编译时间过长"。`.mlpackage` 每次加载重编译（分钟）；`.mlmodelc` 缓存编译资产→首载分钟、后续"几秒"。
3. 质量影响：权重 **palettization 推荐 6-bit**（差异"与 fp16-vs-fp32 相当"）；**Mixed-Bit Palettization** 压 SDXL UNet 至 ~4.04-bit 均值而保 **~73 dB PSNR**（vs fp16 82 dB，线性 8-bit 降至 ~65 dB）；A17 Pro/M4 支持 W8A8 激活量化。无 FID/FVD。
4. 工具/URL：https://github.com/apple/ml-stable-diffusion ；`coremltools`；DiffusionKit https://github.com/argmaxinc/DiffusionKit （Core ML + MLX 后端）。
5. 设备：MPS ✅（A14+/M1+）· 消费NV ❌ · NPU ❌。
6. 注意：**静态形状**——分辨率变需重转（或 `EnumeratedShapes` 分叉）；`SPLIT_EINSUM_V2` 编译长；SD3 MMDiT 需 fp32 + CPU_AND_GPU。**视频扩散 on Core ML：未报告**——Core ML 扩散路径仅图像（SD/SDXL/SD3），无原生视频 DiT Core ML 基准。

**7.5 MLX `mx.compile` + `mx.fast.*`（Apple Silicon, Metal）**
1. 原理：MLX 经 `@mx.compile` 融合逐元/归约为单一 Metal kernel，并提供手写融合原语（`mx.fast.scaled_dot_product_attention` Flash 式、`layer_norm`、`rms_norm`、`rope`），利用分块/共享内存。
2. 加速/显存（M2 Max, MLX 0.31.1）：逐元内存受限最高 **17×（GELU 4k×4k：12ms→0.7ms）**、典型 ~4×；`mx.fast.layer_norm` **2–7× vs eager / ~2–4× vs compile**；`mx.fast.scaled_dot_product_attention` **~1.4–1.5× 快**且避免实体化 T=8192 时 **4.3 GB** 的 T×T 注意力矩阵（内存收益为主）；整训练步 **~1.2×**；推理 `mx.compile` 单独仅 **1.04–1.10×/token**——推理真收益是 `mx.fast` SDPA 而非 compile。冷启动低（lazy/动态图，"改参数形状不触发慢编译"）。
3. 质量影响：融合位等价（`max|python−metal| = 2.4e-7`）。
4. 工具/URL：https://github.com/ml-explore/mlx ；kernel-fusion 基准 https://nipunbatra.github.io/blog/posts/2026-04-25-mlx-kernel-fusion.html ；MFLUX https://github.com/mflux-community/mflux （MLX 原生 FLUX/图像+视频）；MLX Apple-Silicon 基准 arXiv https://arxiv.org/html/2510.18921 。
5. 设备：MPS ✅（Apple Silicon；MLX 亦有 Linux CUDA 后端但以 Apple 为主）· 消费NV ❌ · NPU ❌。
6. 注意：compile **不重写 matmul**——matmul 受限网络收益小（先 profile）；避免依赖输入*值*的 Python 控制流（形状/dtype 可）；声明 `inputs=`/`outputs=` 否则状态快照陈旧；形状变需重编译除非 `shapeless=True`。**无 MLX 视频扩散编译加速数**——MLX 视频刚起步（MFLUX 现"图像与视频模型"，`mlx-teacache` 移植 TeaCache for FLUX），无 compile 专属视频数。

**7 跨技术小结表**

| 技术 | 稳态加速 | 冷启动 | 静态形状痛点 | 设备 |
|---|---|---|---|---|
| `torch.compile`(inductor) | ~1.5×(FLUX)；20–40% 视频 | 67s 全 / 9.6s 区域(2.4s 暖) | 高(`dynamic=True`) | 消费NV, NPU(Jetson) |
| CUDA Graphs(reduce-overhead) | 并入 20–40% 视频 | 捕获前预热 | 很高 | 消费NV, NPU(Jetson) |
| TensorRT(FP8) | 最高 2.5×；60% 延迟(Firefly) | 离线构建+校准 | 高(profiles) | 消费NV, NPU(Jetson) |
| Core ML | 1.11 iter/s SDXL@M2 Ultra | 分钟(`.mlmodelc`→秒) | 很高(静态) | MPS |
| MLX `mx.compile`/`mx.fast` | 1.2× 训练；1.04–1.10× 推理；逐元最高 17× | 低(lazy/动态) | 低(`shapeless=True`) | MPS |

---

## 8. ComfyUI 生态

ComfyUI 是开源视频 DiT（Wan 2.1/2.2、HunyuanVideo、CogVideoX、LTX-Video、Mochi、Cosmos）主流推理前端。加速栈两层：**原生核心**（显存管理、卸载、缓存、原生 FP8 量化、内置 `--use-sage-attention`）与**自定义节点**（TeaCache、GGUF loader、tile VAE、Kijai 封装节点内含 sageattn+TeaCache+block-swap+context-windowing）。Apple Silicon 上 ComfyUI 走 PyTorch **MPS（Metal）**后端（非 MLX）；多项 CUDA/Triton-only 技术（SageAttention）不适用于 MPS。

**8.1 DynamicVRAM（comfy-aimdo, 原生显存管理）**
1. 原理：读 NVML/CUDA 显存压力信号动态决定权重驻留 vs 卸载，替旧静态估算 loader。
2. 加速/显存：无单一加速数（为 enabler，让多 GB 视频 DiT 跑于小显存）；PyTorch ≥2.8 且 NVIDIA 自动启用；调参 `--vram-headroom <GB>`、`--reserve-vram <GB>`、`--disable-nvml-pressure`。
3. 质量影响：无（仅显存管理）。
4. 工具/URL：ComfyUI core `comfy_aimdo`/`main.py`——https://github.com/comfyanonymous/ComfyUI ；flags https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/cli_args.py 。
5. 设备：MPS ❌（未自动启用，回退旧 ModelPatcher，告警"VRAM 估算可能不可靠"）· 消费NV ✅（主，自动开）· NPU 未自动启用。

**8.2 模型卸载 / lowvram / async offload / block-swap**
1. 原理：权重按块在 CPU RAM↔VRAM 间移动（block swap），**异步预取流**使权重传输与计算重叠，仅活跃块驻 GPU。
2. 加速/显存（Kijai WanVideoWrapper, 消费 NVIDIA）：14B Wan T2V @512×512×81 **~16 GB VRAM（20/40 块卸载）**；1.3B Wan T2V，1025 帧经上下文窗（81 帧, 16 重叠）**<5 GB VRAM, RTX 5090 上 10 min**。flags：`--async-offload <NUM_STREAMS>`（默认 2，**NVIDIA 默认开**）、`--disable-async-offload`、`--disable-smart-memory`、`--fast-disk`、`--cpu-vae`、`--cache-ram`。注：`--lowvram` 在 DynamicVRAM 启用时**为 no-op**；LoRA 权重现为 module buffer 遵守 block swap（每 1 GB 未合并 LoRA ≈ +2 卸载块）。
3. 质量影响：无（显存/速度，block swap 付延迟不付输出）。
4. 工具/URL：https://github.com/kijai/ComfyUI-WanVideoWrapper ；flags 同 8.1。
5. 设备：MPS（旧卸载，DynamicVRAM 关）· 消费NV ✅（async-offload 默认开）· NPU（Ascend `torch_npu`/Cambricon `torch_mlu` 手装，卸载可用，async 行为未报告）。

**8.3 批/帧处理 + RAM 压力缓存**
1. 原理：视频潜变量带帧维，批/帧线性放大显存；部分图重执行与 RAM 压力缓存避免跨运行重算未变节点。
2. 加速/显存：`--cache-ram` 默认（active=10% RAM, min 2/max 10 GB；inactive=100% RAM, max 128 GB）；备选 `--cache-lru N`、`--cache-none`、`--cache-classic`、`--highvram`；上下文窗节点分块（Kijai：1025 帧→窗 81+重叠 16→<5 GB on 5090）；UI Batch count 默认 100。
3. 质量影响：无。
4. 工具/URL：cli_args.py（`--cache-*`）；Kijai `context_windows/`——https://github.com/kijai/ComfyUI-WanVideoWrapper 。
5. 设备：MPS ✅ · 消费NV ✅ · NPU ✅。

**8.4 TeaCache 节点（时间步嵌入感知缓存）** *与 §2 TeaCache 同技术，ComfyUI 集成视角*
1. 原理：训练无关，在任意时间步若"时间步嵌入调制的输入差异"小则复用缓存 transformer 输出跳过全 DiT 前向。
2. 加速/显存：最高 **4.41×** Open-Sora-Plan（99.65s→22.62s）；逐模型 **HunyuanVideo 1.6×/2.1×**、Wan2.1 T2V 1.4/1.8/2.0×、**Wan2.1 I2V 最高 2.9×**、CogVideoX1.5 1.3/1.8/2.1×、LTX-Video 1.6/2.1×、Mochi 1.5/2.1×、Cosmos 1.4/2.0×。ComfyUI 调参（Kijai）：新版阈值**高 10×**，用 coefficients 时 **0.25–0.30 范围较好**，`start_step` 可为 0（激进阈值后启动避免早期运动损坏）。
3. 质量影响：4.41× 时 **−0.07% VBench**（可忽略）。
4. 工具/URL：论文 https://arxiv.org/abs/2411.19108 ；项目 https://liewfeng.github.io/TeaCache/ ；代码 https://github.com/ali-vilab/TeaCache 。节点：`ComfyUI-TeaCache`(YunjieYu)、`ComfyUI-WanVideoWrapper`(kijai, TeaCache4Wan2.1)、`ComfyUI-HunyuanVideoWrapper`(kijai)、`ComfyUI-TeaCacheHunyuanVideo`(facok)。
5. 设备：MPS ✅ · 消费NV ✅ · NPU ✅（设备无关）。

**8.5 SageAttention 节点（量化注意力）** *与 §3B 同技术，ComfyUI 集成视角*
1. 原理：即插即用量化注意力替换 DiT 内 `scaled_dot_product_attention`——INT8(V1) 或 INT4(V2) 做 QKᵀ（K 异常平滑）、FP16/FP8 做 PV（FP16/两级累加）。
2. 加速/显存：**V1**：~2.1× over FA2、~2.7× over xformers(OPS)；2.83× 平均实测；CogVideoX 实测 **2.01× on RTX 4090**（163.4→327.6 TOPS）。**V2**：~3× FA2 / ~4.5× xformers on 4090；Hopper 与 FA3-FP8 同速且精度更高。RTX 5090 **560 T, 2.7× FA2**。端到端 CogVideoX1.5-5B(H20)：FA2 **25′34″ → SageAttention 12′07″**（≈2.1×）。
3. 质量影响：可忽略——CogVideoX 端到端 CLIPSIM 0.1837→0.1836、CLIP-T 0.9976→0.9976、VQA-a 68.962→68.839、VQA-t 75.925→75.037、**FScore 3.7684→3.8339**（噪声内）；LLM/图像/视频平均 0.2% 降。
4. 工具/URL：代码 https://github.com/thu-ml/SageAttention ；论文 V1 https://arxiv.org/abs/2410.02367 、V2 https://arxiv.org/abs/2411.10958 、V3 https://arxiv.org/abs/2505.11594 、V2++ https://arxiv.org/abs/2505.21136 。**ComfyUI 原生经 `--use-sage-attention` CLI flag 集成**（cli_args.py）+ Kijai `ComfyUI-WanVideoWrapper` 内含 `ultravico/sageattn`。需 `pip install sageattention` + Triton + CUDA（Ampere≥12.0, Ada fp8≥12.4, Hopper fp8≥12.3, Blackwell/SA2++≥12.8）。
5. 设备：MPS ❌（CUDA+Triton，无 Metal 后端）· 消费NV ✅（Ampere/Ada/Hopper/Blackwell）· NPU ❌。

**8.6 GGUF loader — ComfyUI-GGUF（city96）** *与 §5.2 同技术，ComfyUI 集成视角*
1. 原理：以 llama.cpp GGUF 变比特率亚 8-bit 量化存 DiT/transformer 权重（亦量化 T5），让大视频 DiT 装入低显存。
2. 加速/显存：变比特率量化削占用（主效益 fit 而非速度），README 无加速数。ComfyUI core 明确警告（main.py）**原生 FP8 格式即使大于内存也比 GGUF 快**，建议保 DynamicVRAM + 用原生格式而非 GGUF；ComfyUI 另有原生 `QuantizedTensor`/`TensorCoreFP8Layout`（逐层混精+激活校准）为更快替代。
3. 质量影响：DiT/transformer"受量化影响小于卷积 UNet"；**无 FID/FVD/VBench**。
4. 工具/URL：https://github.com/city96/ComfyUI-GGUF ；节点 `Unet Loader (GGUF)`、`*CLIPLoader (gguf)`。
5. 设备：MPS ⚠（可用，macOS Sequoia 需 torch 2.4.1）· 消费NV ✅ · NPU 未报告。

**8.7 Tile VAE 节点 — VAEDecodeTiled / VAEEncodeTiled（内置）** *与 §4.1/4.2 同技术，ComfyUI 集成视角*
1. 原理：按空间块+时间帧块带重叠解码/编码，避免视频 VAE 解码显存峰值。
2. 加速/显存：内置 `VAEDecodeTiled`——`tile_size`(默认 512, 64–4096, step 32)、`overlap`(默认 64)、`temporal_size`(默认 64 帧, 8–4096, step 4)、`temporal_overlap`(默认 8, 4–4096, step 4)；自动钳位（overlap→tile_size/4 if tile<4×overlap 等）。效益为 VRAM-fit 非 speed（分块解码通常更慢），无加速数。
3. 质量影响：块缝由 overlap 缓解（默认 64 空间 / 8 时间）。
4. 工具/URL：https://docs.comfy.org/built-in-nodes/VAEDecodeTiled.md 、`VAEEncodeTiled.md`（内置，无需自定义安装）。
5. 设备：MPS ✅ · 消费NV ✅ · NPU ✅。

**MPS 诚实缺口：** ComfyUI on Apple Silicon 走 PyTorch MPS/Metal 后端（M1–M4, macOS 13+）。生态中最大的实测推理加速（SageAttention、DynamicVRAM async-offload）为 CUDA/Triton/NVML 绑定，**不适用 MPS**。MPS 可用加速为 **TeaCache**（设备无关，主实测收益 1.3–4.4×）、**tile VAE** 与 **GGUF/卸载**（VRAM-fit，无发布 MPS 速度数）。除 TeaCache 设备无关数外，MPS 速度比为 **"未报告"**。

---

# SECTION A — 综合总表

> 列：技术 | 加速比/省显存 | 质量影响 | 训练侧/推理侧 | 适用设备(MPS/消费NV/NPU)。`【S#】` 标所属原始 section。✅ 已测/原生；△ 可移植无实测；❌ 不支持/未报告。"未报"=未检索到数据。

| 技术 | 加速比/省显存 | 质量影响 | 训练/推理 | 设备(MPS/消费NV/NPU) |
|---|---|---|---|---|
| 【S1】Consistency Models | 单步；FID 3.55/6.20 | 单步 FID 3.55/6.20 | 训练侧+推理侧 | ❌/✅/❌ |
| 【S1】LCM | 1步FID80.0→4步21.9；7×/8GB(PIXART-δ) | 1步质量降 | 训练侧+推理侧 | ❌/✅/❌ |
| 【S1】DMD | FID 2.62/11.49；20 FPS | FID 2.62/11.49 | 训练侧+推理侧 | ❌/✅/❌ |
| 【S1】DMD2 | FID 1.28/8.35；500×；超教师 | 超教师 | 训练侧+推理侧 | ❌/✅/❌ |
| 【S1】ADD | 1步0.09s；1/2/4步FID19.7→20.3；4步ELO胜SDXL-50步 | 4步胜50步 | 训练侧+推理侧 | ❌/✅/❌ |
| 【S1】PCM | 1–16步；优于LCM | 优于LCM(定性) | 训练侧+推理侧 | ❌/✅/❌ |
| 【S1】SDXL-Lightning | 1/2/4/8步1024px表 | 步进表见原section | 训练侧+推理侧 | ❌/✅/❌ |
| 【S1】LADD/SD3-Turbo | 未报 | 未报 | 训练侧+推理侧 | 未报 |
| 【S1】CoDMD | Wan 50→4步 ~25×；VBench 84.5–84.9 | 优于DMD/rCM | 训练侧+推理侧 | ❌/✅/❌ |
| 【S1】Dynamic-in-Few-Step | Wan-14B 30× over 50步 | 定性 | 训练侧+推理侧 | ❌/✅/❌ |
| 【S1】WanToFight | 30 FPS on RTX 5090 | 未报 | 训练侧+推理侧 | ❌/✅/❌ |
| 【S1】Causal Forcing++ | 帧级2步；+0.335 VisionReward | +0.335 VR | 训练侧+推理侧 | ❌/✅/❌ |
| 【S1】AnyFlow/PDD/SGMD/DFD/DUET/VideoLCM | 未报(VideoLCM FVD未报) | 未报 | 训练侧+推理侧 | 未报 |
| 【S2】TeaCache | 4.41×–4.91×(OSP)；Wan1.4–2.9×、Hunyuan1.6–2.1×、Cog1.3–2.1×等 | −0.07% VBench(可忽略) | 推理侧(训练无关) | ✅/✅/✅ |
| 【S2】Δ-DiT | 见DuCa对比表(独立数未转录) | 见对比表 | 推理侧 | ✅/✅/❌ |
| 【S2】FasterCache | 未报 | 未报 | 推理侧 | ✅/✅/❌ |
| 【S2】T-GATE | SD-XL 1.68×(1080Ti) | FID不降反升 | 推理侧 | ✅/✅/❌ |
| 【S2】ToCa | 见DuCa对比表 | 见对比表 | 推理侧 | ✅/✅/❌ |
| 【S2】DuCa | Tables II–V横向对比 | 见对比表 | 推理侧 | ✅/✅/❌ |
| 【S2】DeepCache | 未报 | 未报 | 推理侧 | ✅/✅/❌ |
| 【S2】Faster-Diffusion | 未报 | 未报 | 推理侧 | ✅/✅/❌ |
| 【S3】FlashAttention-2 | 显存~10×(seq2K)/~20×(seq4K)；2× over FA1 | 无损失(精确) | 推理侧 | ❌/✅/❌ |
| 【S3】FlashAttention-3 | 1.5–2.0× over FA2(H100)；740TFLOPs；FP8 1.2PFLOPs | FP8误差比朴素低2.6× | 推理侧 | ❌/数据中心NV/❌ |
| 【S3】FlashAttention-4 | 未报(新发布) | 未报 | 推理侧 | ❌/✅(Blackwell)/❌ |
| 【S3】FlashAttention/MPS | 无(MPS无融合kernel) | 精确但慢 | 推理侧 | ✅(MLX替代)/n/a/n/a |
| 【S3】SageAttention v1 | 2–5× over FA；H20 CogVideoX 25′34″→12′07″(2.1×)；5090 560T 2.7× | 端到端可忽略 | 推理侧(PTQ) | ❌/✅/❌ |
| 【S3】SageAttention2 | 3× FA2/4.5× xformers(4090) | 可忽略(精度敏感推荐) | 推理侧(PTQ) | ❌/✅/❌ |
| 【S3】SageAttention2++ | 3.9× over FA | 同SA2 | 推理侧(PTQ) | ❌/✅/❌ |
| 【S3】SageAttention3 | 1038 TOPS(5090) 5× over FA | FP4精度较低 | 推理侧(PTQ) | ❌/✅(Blackwell)/❌ |
| 【S3】Sliding Tile Attention | 注意力2.8–17× FA2；Hunyuan 945→685s(tf)→268s(ft) | tf无降；ft 0.09% VBench | 推理侧(ft需训练) | ❌/✅/❌ |
| 【S3】SpargeAttn | 单一×未报(随稀疏度可调) | 不牺牲端到端 | 推理侧(tf)+训练(SA2) | ❌/✅/❌ |
| 【S3】NATTEN | Hopper/Blackwell FNA(按FLOPs缩减) | 局部性丢全局；VBench未报 | 推理侧 | ❌/✅/❌ |
| 【S3】3D注意力分解 | 渐近FLOPs降 | 显著变差(FVD升,易崩溃) | 架构级(训练侧) | △/✅/△(质量损) |
| 【S3】SANA-Video | 16× vs Wan1.3B；NVFP4 2.4×(71s→29s) | VBench相当 | 训练侧+推理侧 | △/✅/△ |
| 【S3】SANA-Video 2.0 | VBench84.30；720p/60s fwd 3.2×；单H100 120× vs Wan2.2 | VBench84.30 | 训练侧+推理侧 | △/✅/△ |
| 【S3】ARL2 | 2.26× wall-clock；54%显存降 | 质量相当 | 训练侧+推理侧 | △/✅/△ |
| 【S3】SALAD | 90%稀疏；1.52–2.03× | 与全当相当 | 训练侧+推理侧 | △/✅/△ |
| 【S3】HLA | 未报(用于大型视频DiT) | 未报 | 训练侧+推理侧 | △/✅/△ |
| 【S4】空间分块VAE | 3070 56.3%→16.0%(≈3.5×)；延迟+77%/63% | 边界淡网格(overlap≥64不可见) | 推理侧 | ✅/✅/✅ |
| 【S4】时间分块VAE | Hunyuan 32GB→8GB(4×)；CogX 19→11GB | 边界闪烁(overlap过小) | 推理侧 | ✅/✅/✅ |
| 【S4】流式VAE(CogVideoX) | 49帧@720×480 19→11GB；顺序<4GB | 近无损重建 | 推理侧 | ✅/✅/✅ |
| 【S4】VAE INT8/FP8量化 | torchao 2.52→2.43s；TRT SDXL 1.72×/1.95×；Firefly视频60%延迟/2.5×骨干 | 图像"几乎一致"；视频FID/FVD未报 | 校准侧+推理侧 | △/✅/✅(INT8) |
| 【S4】VAE卸载CPU/远端 | CogX 33→19GB；顺序<4GB；remote_decode远端 | 无损失 | 推理侧 | ✅/✅/✅ |
| 【S5】HunyuanVideo FP8 QDQ | 省~10GB,无加速 | 未报 | 校准侧+推理侧 | ❌/✅/❌ |
| 【S5】GGUF Q4/Q8(ComfyUI-GGUF) | HunyuanI2V 25.6→7.88GB；Wan14B 29.1→10.1GB | 无正式FID/VBench | 推理侧(PTQ) | ⚠(torch2.4.1)/✅/❌ |
| 【S5】Jetson INT8(TensorRT) | Orin Nano 9×延迟；TRT拒transformer层 | 未报 | 校准侧+推理侧 | ❌/✅(Jetson)/△ |
| 【S5】HiF8(Ascend/Wan2.1) | 未报具体× | 5维VBench全≥BF16 | 校准侧+推理侧 | ❌/❌/✅(Ascend) |
| 【S5】SmoothQuant/AWQ系(SVDQuant等) | SVDQuant NVFP4 3.1×(5090)；余见原section | 接近全精 | 校准侧+推理侧 | △/✅/△ |
| 【S5】6Bit-Diffusion | NVFP4/INT8动态 1.92×/3.32× | 接近全精 | 推理侧(PTQ) | △/✅(Blackwell)/△ |
| 【S5】CineMobile | MediaTek NPU <1GB | 未报 | 校准侧+推理侧 | ❌/❌/✅(MediaTek) |
| 【S5】AdaTSQ/OrbitQuant | 未报 | 未报 | 校准侧+推理侧 | △/✅/△ |
| 【S6】Diffusion Forcing | FVD未报 | 未报 | 训练侧+推理侧 | 未报 |
| 【S6】CogVideoX Frame Pack | 3–4×速度/3×显存(总体) | 3D因果VAE近无损 | 训练侧+推理侧 | ✅/✅/✅ |
| 【S6】Self-Forcing | 未报 | 未报 | 训练侧+推理侧 | 未报 |
| 【S6】Flex-Forcing | FPS/VBench图示(未转录) | 未报 | 训练侧+推理侧 | 未报 |
| 【S6】Forcing-KV/Focused Forcing | 未报 | 未报 | 训练侧+推理侧 | 未报 |
| 【S6】LongLive | VBench未报 | 未报 | 训练侧+推理侧 | 未报 |
| 【S6】MiniWorld | VBench未报 | 未报 | 训练侧+推理侧 | 未报 |
| 【S6】NVFP4 KV cache | 未报 | 未报 | 训练侧+推理侧 | ❌/✅(Blackwell)/❌ |
| 【S7】torch.compile | FLUX 1.5×；视频20–40%；冷启67.4s/区域9.6s | 无回归(FLUX) | 推理侧 | 有限/✅/✅(Jetson) |
| 【S7】CUDA Graphs | 并入20–40%视频(DenseNet121示意2.05×) | 无(位精确) | 推理侧 | ❌/✅/✅(Jetson) |
| 【S7】TensorRT(video) | Firefly 60%延迟/2.5×骨干 | 未数值(无FID/FVD) | 编译侧+推理侧 | ❌/✅/✅(Jetson) |
| 【S7】Core ML | SDXL M2Ultra 1.11iter/s；iPhone14PM 7.9s | 6-bit palett ~73dB PSNR | 编译侧+推理侧 | ✅/❌/❌ |
| 【S7】MLX mx.compile/mx.fast | 逐元最高17×；SDPA 1.4–1.5×；推理1.04–1.10×；训练1.2× | 位等价(2.4e-7) | 推理侧 | ✅/❌/❌ |
| 【S8】DynamicVRAM | enabler(无单一×) | 无 | 推理侧 | ❌/✅/❌ |
| 【S8】async offload/block-swap | 14B Wan ~16GB(20/40块)；1.3B 1025帧<5GB/10min(5090) | 无 | 推理侧 | 旧/✅/△ |
| 【S8】RAM压力缓存 | 默认cache-ram(10%/100%RAM) | 无 | 推理侧 | ✅/✅/✅ |

---

# SECTION B — 可组合加速配方（Composable Recipes）

> 各配方标注**实测组件**与**组合估算（ESTIMATE，未整体实测）**。多技术叠加不简单相乘——存在瓶颈转移（如步数蒸馏后注意力占比下降、Sage 收益随之降低）与作用域重叠（STA 与 SA3 都改注意力）。估算已保守并标注。

### 配方 1 — TeaCache + SageAttention2 + 步数蒸馏(CoDMD) + VAE 时间分块
- **适用设备：** 消费NV（RTX 4090/5090, H20）
- **原理 / 正交性：** 步数蒸馏把 50 步降到 4 步（CoDMD 实测 ~25×）；TeaCache 跳过冗余前向（实测 ~2×）；SageAttention2 加速剩余注意力（INT4 QK+FP8 PV，注意力约占 ~85% 算力 → 端到端 ~1.7–2×）；VAE 时间分块控制解码显存（HunyuanVideo 32GB→8GB）。四者作用于不同环节：步数、每步前向次数、每步注意力、解码显存，正交可叠。
- **实测组件：** CoDMD 25×（步数）；TeaCache ~2×；SageAttention ~2×（端到端）；VAE 时间分块 4× 省显存。
- **组合估算（ESTIMATE）：** 25× × 2× × ~1.8× ≈ 理论 ~90× over 50 步基线；考虑瓶颈转移（4 步后注意力占比下降、Sage 收益打折）与解码/非注意力开销，**保守端到端 30–60×**。显存：VAE 段 32GB→8GB。
- **风险：** 蒸馏需服务端训练（CoDMD 训练成本高）；TeaCache 阈值过高致早期运动损坏（Kijai 建议 0.25–0.30，后启动）；Sage2 需 CUDA≥12.4/Ada+。

### 配方 2 — Sliding Tile Attention + SageAttention3(FP4) + torch.compile(区域编译) + FP8
- **适用设备：** 消费NV Blackwell（RTX 5090）/ B200
- **原理 / 正交性：** 注意力是视频 DiT 主导开销（5s 720P 约 800/945s）。STA 把全注意力变 3D 滑窗（注意力 kernel 2.8–17×，Hunyuan 945→685s training-free）；SA3 用 FP4 Tensor Core（1038 TOPS, 5× over FA on 5090）；torch.compile 区域编译稳态 1.5× 且冷启动 9.6s；FP8 权重省显存。STA 与 SA3 都作用于注意力（部分重叠，取较强者 + 部分叠加）。
- **实测组件：** STA Hunyuan 945→685s(tf)/268s(ft, 0.09% VBench)；SA3 5× over FA(5090)；torch.compile 1.5×。
- **组合估算（ESTIMATE）：** 注意力部分 STA×SA3 叠加 ≈ 注意力 5–10×；端到端注意力占 ~85% → 端到端 ~3–5×；+compile 1.5× → **端到端 ~5–7×**。Hunyuan 945s → 估 ~150–250s（tf 路径叠 SA3/compile）。
- **推荐实测最优：** STA finetune 版 268s（0.09% VBench）已是已测最优单技术；叠加 SA3/compile 需实测验证。
- **风险：** FP4(SA3) 精度低于 INT4，精度敏感场景改用 SA2/SA2++；动态形状需 `dynamic=True`+`mark_dynamic`。

### 配方 3 — GGUF Q4_K_M + block-swap async offload + 时间分块VAE + TeaCache
- **适用设备：** 低显存消费NV + MPS（ComfyUI-GGUF on MPS 需 torch 2.4.1）
- **原理 / 正交性：** GGUF Q4 把 Wan2.1-14B 29.1GB→10.1GB；block-swap（20/40 块）使 14B 在 ~16GB 跑；时间分块 VAE HunyuanVideo 32GB→8GB；TeaCache 设备无关 1.4–2.9×。组合目标：让大模型在小显存跑起来 + 适度加速。四者分别压权重显存、激活显存、解码显存、前向次数。
- **实测组件：** GGUF Q4_K_M Wan14B 29.1→10.1GB；block-swap 14B ~16GB；VAE 时间分块 4× 省显存；TeaCache ~2×。
- **组合估算（ESTIMATE）：** 显存 Wan2.1-14B 29GB → ~10–16GB（measured）；端到端速度 ~2×（主为 cache）。**核心价值是 fit（从跑不动到跑得动）而非极限加速。**
- **风险：** ComfyUI 官方提示原生 FP8 比 GGUF 更快（能用 FP8 优先）；MPS 上 SageAttention 不可用，注意力加速受限；MPS GGUF 需锁定 torch 2.4.1。

### 配方 4 — 线性注意力(SANA-Video 2.0) + NVFP4 + 编译DiT + mx.fast SDPA
- **适用设备：** 消费NV RTX 5090（SANA-Video）/ H100；MLX 路径供 MPS
- **原理 / 正交性：** 线性注意力把 O(N²)→O(N)，SANA-Video 比 Wan 2.1-1.3B 快 16×；NVFP4 进一步 2.4×（71s→29s, 5s 720p）；编译 DiT fwd 3.2×（720p/60s）；mx.fast SDPA 避免实体化 4.3GB 注意力矩阵（内存收益）。SANA-Video 2.0 VBench 84.30。线性注意力需训练/微调（训练侧），mx.fast 为 MPS 原生注意力路径。
- **实测组件：** SANA-Video 16× vs Wan1.3B；NVFP4 2.4×（71→29s）；SANA-Video 2.0 编译 fwd 3.2×、单 H100 120× vs Wan2.2；mx.fast SDPA 1.4–1.5× + 4.3GB 矩阵避免。
- **组合估算（ESTIMATE）：** SANA-Video 2.0 已测单 H100 120× vs Wan2.2-A14B（架构内已含线性+编译）；单卡 5090 路径 16×(vs Wan1.3B) × 2.4×(NVFP4) → **~38× 端到端**（部分已含于 SANA 架构，非纯叠加）。MPS 经 MLX+mx.fast 仅获 1.4–1.5× 注意力 + 内存收益，无线性注意力生产实现（△）。
- **风险：** 线性注意力需重新训练/微调（非即插，迁移成本高）；MPS 无线性注意力融合实现。

### 配方 5 — TensorRT FP8 + torch.compile(区域) + CUDA Graphs + INT8/FP8 VAE
- **适用设备：** 消费NV（Ada/Hopper FP8）+ Jetson NPU（TensorRT 支持 Orin）
- **原理 / 正交性：** TRT FP8 引擎（Adobe Firefly 视频 60% 延迟, 2.5× 骨干）；torch.compile 区域编译 1.5×（冷启动 9.6s）；CUDA Graphs 消 launch 开销（reduce-overhead）；INT8 VAE（torchao）降解码显存。组合 = 编译栈全开，模型无关。注意 TRT 与 torch.compile 一般二选一（都接管图），此处以 TRT 为主、compile/Graphs 用于 TRT 不覆盖的段或预处理/VAE。
- **实测组件：** TRT Firefly 60% 延迟/2.5× 骨干；torch.compile 1.5×；INT8 VAE（图像 torchao 2.52→2.43s）。
- **组合估算（ESTIMATE）：** TRT 2.5× × compile 1.5× → **~3.5× 端到端**（部分重叠，TRT 已含层融合）；Adobe Firefly 实测 60% 延迟 ≈ 2.5× 已含 FP8 引擎。Jetson INT8 路径 9× 延迟但 TRT 拒 transformer 层（已知限制，需 TRT 支持视频 DiT transformer 后方可兑现）。
- **风险：** 动态形状需 TRT optimization profile 或重建；FP8 需 Ada/Hopper+；INT8/FP8 视频侧未单独测；TRT 引擎构建一次性但非平凡。

---

# Sources（去重合并）

> 下列 URL 为各 section agent 实际抓取的一方来源（arXiv、GitHub raw、HuggingFace 文档、DeepWiki、NVIDIA/PyTorch/Apple 官方博客）。已跨 section 去重。Section 1/2/5/6 以覆盖总结交付，其逐条来源 URL 未在本汇总输入完整转录，已注明；此处列出的是被实际转录的具体 URL。

**注意力（Section 3/8）**
- SageAttention 仓库（V1/2/2++/3）：https://github.com/thu-ml/SageAttention
- SageAttention v1（ICLR 2025）：https://arxiv.org/abs/2410.02367
- SageAttention2（ICML 2025, INT4+FP8）：https://arxiv.org/abs/2411.10958
- SageAttention2++（FP8-in-FP16, 3.9×）：https://arxiv.org/abs/2505.21136
- SageAttention3（NeurIPS 2025 Spotlight, FP4 Blackwell, 1038 TOPS/5×）：https://arxiv.org/abs/2505.11594
- SpargeAttn 仓库：https://github.com/thu-ml/SpargeAttn
- SpargeAttn（ICML 2025, 训练无关稀疏）：https://arxiv.org/abs/2502.18137
- SpargeAttention2（2026）：https://arxiv.org/abs/2602.13515
- FlashAttention 仓库（FA-2/3/4, 滑窗, MPS 缺席）：https://github.com/Dao-AILab/flash-attention
- FlashAttention-2：https://arxiv.org/abs/2307.08691
- FlashAttention-1：https://arxiv.org/abs/2205.14135
- FlashAttention-3（1.5–2×, 740TFLOPs, FP8 1.2PFLOPs, 误差低2.6×）：https://arxiv.org/abs/2407.08608
- FA-3 博客：https://tridao.me/blog/2024/flash3
- NATTEN 仓库（Neighborhood Attention, Hopper/Blackwell FNA）：https://github.com/SHI-Labs/NATTEN
- Generalized Neighborhood Attention：https://arxiv.org/abs/2504.16922
- Dilated NA：https://arxiv.org/abs/2209.15001
- Sliding Tile Attention / FastVideo（2.8–17× FA2; Hunyuan 945→685s/268s）：https://arxiv.org/abs/2502.04507
- CogVideoX（3D 全注意胜 2D+1D; FVD 消融）：https://arxiv.org/abs/2408.06072
- HunyuanVideo（3D 全注意, 13B）：https://arxiv.org/abs/2412.03603
- HunyuanVideo 1.5（selective + sliding tile attention, SSTA, 8.3B）：https://arxiv.org/abs/2511.18870
- Wan2.1 仓库（全 3D 自注意 + T5 交叉注意; 非 NATTEN）：https://github.com/Wan-Video/Wan2.1
- Wan2.1 技术报告：https://arxiv.org/abs/2503.20314
- SANA-Video（线性 DiT, 16× vs Wan1.3B, NVFP4 2.4×）：https://arxiv.org/abs/2509.24695
- SANA-Video 2.0（混合线性-softmax, VBench 84.30, 120× vs Wan2.2）：https://arxiv.org/abs/2607.21553
- ARL2（混合线性 AR 视频扩散, 2.26×/54% 显存）：https://arxiv.org/abs/2605.16579
- SALAD（线性+稀疏, 90% 稀疏, 1.52–2.03×）：https://arxiv.org/abs/2601.16515
- HLA（Hadamard Linear Attention）：https://arxiv.org/abs/2602.12128
- PyTorch SDPA 文档 v2.13（优化 kernel 仅 CUDA）：https://pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch MPS 后端说明：https://pytorch.org/docs/2.13/notes/mps.html
- MLX（Apple 原生框架）：https://github.com/ml-explore/mlx
- arXiv API（论文发现）：https://export.arxiv.org/api/query

**缓存（Section 2/8）**
- TeaCache 仓库（ali-vilab）：https://github.com/ali-vilab/TeaCache
- TeaCache 项目页：https://liewfeng.github.io/TeaCache/
- TeaCache 论文（CVPR 2025, −0.07% VBench）：https://arxiv.org/abs/2411.19108
- Δ-DiT：https://arxiv.org/abs/2406.01125
- T-GATE GitHub README（1.68× SD-XL, FID 改善, 1080 Ti）（原 Section 2 抓取，具体 URL 未转录）
- DuCa 全文 HTML（Tables II–V 横向对比）（原 Section 2 抓取，具体 URL 未转录）

**VAE（Section 4/8）**
- 8GB VRAM 跑 Hunyuan — ComfyUI Blog：https://blog.comfy.org/p/running-hunyuan-with-8gb-vram-and
- VAEDecode(Tiled) — ComfyUI Cloud docs：https://comfy.icu/node/VAEDecodeTiled
- ComfyUI-TiledVaeLite README：https://raw.githubusercontent.com/hum-ma/ComfyUI-TiledVaeLite/main/README.md
- AutoencoderKLCogVideoX — HF diffusers：https://huggingface.co/docs/diffusers/v0.36.0/en/api/models/autoencoderkl_cogvideox
- CogVideoX pipeline — HF diffusers：https://huggingface.co/docs/diffusers/v0.32.2/en/api/pipelines/cogvideox
- VAE Hybrid Inference — bytedance/Video-As-Prompt：https://raw.githubusercontent.com/bytedance/Video-As-Prompt/main/diffusers/docs/source/en/hybrid_inference/vae_decode.md
- Accelerating Generative AI Part III — PyTorch blog：https://pytorch.org/blog/accelerating-generative-ai-3/
- NVIDIA TensorRT SD 2× 8-bit PTQ：https://developer.nvidia.com/blog/tensorrt-accelerates-stable-diffusion-nearly-2x-faster-with-8-bit-post-training-quantization/
- NVIDIA TensorRT 视频扩散优化：https://developer.nvidia.com/blog/optimizing-transformer-based-diffusion-models-for-video-generation-with-nvidia-tensorrt/
- WF-VAE 大视频 — DeepWiki：https://deepwiki.com/PKU-YuanGroup/WF-VAE/5.3-working-with-large-videos
- IV-VAE（arXiv:2411.06449）：https://arxiv.org/abs/2411.06449
- HunyuanVideo 1.5 VRAM 需求 — Will It Run AI：https://willitrunai.com/blog/hunyuanvideo-1-5-vram-requirements

**量化（Section 5）**
- SVDQuant / SageAttention / SANA-Streaming / HunyuanVideo FP8 QDQ / HiF8(Ascend) / 6Bit-Diffusion / CineMobile / ViDiT-Q / CLQ / DiRotQ / HyperQuant-LTX2 / Wan2.2 W4A4（原 Section 5 经 arXiv/HF/GitHub/DeepWiki 抓取 25+ 来源，逐条 URL 未在本汇总转录）

**长视频（Section 6）**
- Diffusion Forcing：https://arxiv.org/abs/2407.01392
- CogVideoX：https://arxiv.org/abs/2408.06072
- CogVideoX1.5 模型卡（HF）（原 Section 6 抓取）
- Flex-Forcing 项目页（原 Section 6 抓取）
- Self-Forcing / Forcing-KV / Focused Forcing / LongLive / MiniWorld / NVFP4 KV cache（原 Section 6 经 arXiv 抓取，具体 URL 未转录）

**编译（Section 7）**
- torch.compile + Diffusers DevLog（2026-05-11）：https://docs.pytorch.org/devlogs/inductor/2026-05-11-torch-compile-and-diffusers/
- torch.compile 端到端教程：https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html
- NVIDIA TensorRT 视频扩散博客（2025-04-21）：https://developer.nvidia.com/blog/optimizing-transformer-based-diffusion-models-for-video-generation-with-nvidia-tensorrt/
- HunyuanVideo-1.5 推理加速 — DeepWiki：https://deepwiki.com/Tencent-Hunyuan/HunyuanVideo-1.5/6.1-inference-acceleration
- ComfyUI-KJNodes Torch 编译系统 — DeepWiki：https://deepwiki.com/kijai/ComfyUI-KJNodes/7.2-torch-compilation-system
- Stable Diffusion with Core ML — Apple ML Research：https://machinelearning.apple.com/research/stable-diffusion-coreml-apple-silicon
- apple/ml-stable-diffusion — GitHub：https://github.com/apple/ml-stable-diffusion
- MLX kernel fusion 博客（2026-04-25）：https://nipunbatra.github.io/blog/posts/2026-04-25-mlx-kernel-fusion.html
- MLX — GitHub：https://github.com/ml-explore/mlx
- MLX Apple-Silicon 基准（arXiv:2510.18921）：https://arxiv.org/html/2510.18921
- mflux-community/mflux — GitHub：https://github.com/mflux-community/mflux
- argmaxinc/DiffusionKit — GitHub：https://github.com/argmaxinc/DiffusionKit

**ComfyUI 生态（Section 8）**
- ComfyUI 仓库：https://github.com/comfyanonymous/ComfyUI
- ComfyUI main.py（raw）：https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/main.py
- ComfyUI cli_args.py（raw）：https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/cli_args.py
- ComfyUI QUANTIZATION.md（raw）：https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/QUANTIZATION.md
- ComfyUI docs llms.txt：https://docs.comfy.org/llms.txt
- Comfy Settings：https://docs.comfy.org/interface/settings/comfy.md
- VAEDecodeTiled 文档：https://docs.comfy.org/built-in-nodes/VAEDecodeTiled.md
- 系统需求：https://docs.comfy.org/installation/system_requirements.md
- macOS Desktop：https://docs.comfy.org/installation/desktop/macos.md
- ComfyUI-GGUF（city96）：https://github.com/city96/ComfyUI-GGUF
- ComfyUI-WanVideoWrapper（kijai）：https://github.com/kijai/ComfyUI-WanVideoWrapper

**蒸馏（Section 1）**
- Consistency Models / LCM / DMD / DMD2 / ADD / PCM / SDXL-Lightning / LADD / SD3-Turbo / CoDMD / Dynamic-in-Few-Step / WanToFight / Causal Forcing++ / AnyFlow / PDD / SGMD / DFD / DUET / VideoLCM（原 Section 1 经 arXiv web_fetch 抓取摘要/全文 HTML，逐条 URL 未在本汇总输入转录；ADD 与 SDXL-Lightning 步进 FID/CLIP 表从全文 HTML 直接提取）

---

*报告完。所有数字均来自上述被抓取的一方来源；未检索到数据处已如实标注"未检索到数据/未报"。组合配方中的端到端倍数为估算（ESTIMATE），已与实测组件明确区分。*
