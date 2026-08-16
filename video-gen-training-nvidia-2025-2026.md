# 视频生成模型在 NVIDIA 数据中心集群上的工业级训练方案调研报告

> 调研范围：训练侧（NVIDIA 数据中心）。调研日期：2026-08-13。数据来源：web_fetch 实际抓取的官方 model card / GitHub / arXiv / NVIDIA 官网（URL 随文标注）。
> 关注点：训练侧选择如何影响下游异构端侧推理（Apple Silicon / 消费级 NVIDIA GPU / 工业边缘 NPU）的鲁棒性。

> 说明：本环境的 `web_search` API 未配置，故以 `web_fetch` 直接抓取权威页面（HuggingFace、GitHub、arXiv、NVIDIA 官网、Wikipedia）作为一手来源。MiniMax Video-01 / Mochi-1 的 HF model card 为受限页面（401），改用其 GitHub 仓库。

---

## 1. 主流开源视频生成模型与架构

当前开源视频生成已全面收敛到 **Diffusion Transformer (DiT) + Flow Matching + 3D 时空 VAE** 范式，差异主要在参数量、VAE 压缩比、文本编码器与任务覆盖（T2V / I2V）。

### 模型汇总表

| 模型 (机构) | 发布 | 架构 | 参数量 | VAE 类型 (T×H×W 压缩) | 文本编码器 | 分辨率/时长/帧率 | 任务 | 开源协议 | 权重 |
|---|---|---|---|---|---|---|---|---|---|
| **Wan 2.1** (阿里) | 2025-02 | DiT + Flow Matching | 1.3B / 14B | Wan-VAE (3D causal 时空) | T5 (多语言, umt5-xxl) | 480P/720P, ~5s (81帧), 16fps | T2V/I2V/FLF2V/VACE/T2I | Apache 2.0 | ✅ |
| **Wan 2.2** (阿里) | 2025-07 | **MoE DiT** (双专家) | A14B(27B总/14B激活) / TI2V-5B(dense) | Wan2.2-VAE (16×16×4=64×, patchify 后 4×32×32) | T5 (多语言) | 720P, 24fps; 480P/720P | T2V/I2V/TI2V/S2V/Animate | Apache 2.0 | ✅ |
| **HunyuanVideo** (腾讯) | 2024-12 | DiT (dual-stream→single-stream, full attn) | 13B | Causal 3D VAE (4×8×16) | **MLLM** (decoder-only + 双向 token refiner) | 720p(1280×720), 129帧, 5s | T2V (I2V 计划中) | tencent-hunyuan-community | ✅ (含 FP8) |
| **CogVideoX / 1.5** (智谱) | 2024-08/11 | Expert Transformer (3d_rope) | 2B / 5B | 3D Causal VAE | T5 (英文) | 480p~1360×768, 5–10s, 8–16fps | T2V/I2V/V2V | 2B=Apache2.0; 5B=CogVideoX License | ✅ |
| **LTX-Video / LTX-2** (Lightricks) | 2024→2025 | DiT | 2B / 13B | (高压缩 VAE) | (T5) | 1216×704 实时 30fps; <720×1280 | I2V 为主 | other (商用受限) | ✅ |
| **Mochi 1** (Genmo) | 2024-10 | AsymmDiT | 10B | AsymmVAE (8×8×6, 128× 压缩, 12ch) | T5-XXL (单编码器) | 480p(848×480), 31帧 | T2V | Apache 2.0 | ✅ |
| **Allegro** (Rhymes AI) | 2024-10 | DiT (基于 Open-Sora-Plan) | DiT 2.8B / VAE 175M | 3D VAE | T5 | 720×1280, 88帧, 6s@15fps | T2V / TI2V | Apache 2.0 | ✅ |
| **Open-Sora 2.0** (HPC-AI Tech) | 2025-03 | MMDiT (dual+single stream, 3D RoPE) | 11B (Flux 初始化) | Video DC-AE (4×32×32, 高压缩) | T5-XXL + CLIP-Large | 256px/768px, 128帧, 5s@24fps | T2V/I2V | Apache 2.0 | ✅ |
| **Open-Sora 1.3** (HPC-AI Tech) | 2025-02 | DiT | 1B | 统一时空 VAE | — | 720×1280, 5s | T2V/I2V | Apache 2.0 | ✅ |
| **Pyramid Flow** (Kuaishou/HKUST) | 2024-10 | DiT + 金字塔流匹配 (多分辨率) | ~2B | 3D VAE | T5 | 768p, 24fps, ≤10s | T2V | 开源 (MIT 系) | ✅ |
| **MiniMax Video-01 / Hailuo** | 2024 | (闭源系) | 未公开 | — | — | 1080p, ~6s | T2V/I2V | 权重受限/无训练代码 | ⚠️ 半闭源 |
| Step-Video-T2V (StepFun) | 2025-02 | DiT | 30B | StepVideo VAE (8×16×16) | — | 1080p | T2V | 部分开源 | ⚠️ |

来源：
- Wan 2.1/2.2：https://github.com/Wan-Video/Wan2.2 · https://huggingface.co/Wan-AI/Wan2.1-T2V-14B · arXiv 2503.20314
- HunyuanVideo：https://huggingface.co/tencent/HunyuanVideo · arXiv 2412.03603
- CogVideoX：https://github.com/THUDM/CogVideo · arXiv 2408.06072
- LTX-Video：https://huggingface.co/Lightricks/LTX-Video
- Mochi 1：https://github.com/genmoai/mochi
- Allegro：https://github.com/rhymes-ai/Allegro · arXiv 2410.15458
- Open-Sora 2.0：https://github.com/hpcaitech/Open-Sora · arXiv 2503.09642

**关键架构观察**
- **MoE 成为视频 DiT 新趋势**：Wan 2.2 的 A14B 用"高噪声专家 + 低噪声专家"按 SNR 阈值切换，27B 总参数但每步仅 14B 激活，推理算力/显存几乎不变却扩容模型容量。
- **高压缩 VAE 成为降本关键**：Open-Sora 2.0 的 Video DC-AE（4×32×32）将 768p 5s 视频的 token 数从 76K 降到 19K（4×），训练吞吐 5.2×、推理 10×+；Wan 2.2-VAE 同样走 4×32×32（含 patchify）。代价：重建质量略降 + 扩散模型适配变难。
- **文本编码器分化**：T5-XXL（Wan/Mochi/Open-Sora/CogVideoX）vs MLLM decoder-only（HunyuanVideo）。T5-XXL 约 5B 参数、20GB+，是端侧部署的主要负担。

---

## 2. 训练硬件（Hopper → Blackwell → Blackwell Ultra）

### GPU 规格表

| GPU | 架构 | 显存 | 显存带宽 | TDP (SXM) | Tensor Core | 关键低精度 | NVLink |
|---|---|---|---|---|---|---|---|
| H100 SXM | Hopper | 80GB HBM3 | 3.35 TB/s | 700W | 4th gen | FP8 (TE 1st gen) | NVLink-3 900 GB/s |
| H200 SXM | Hopper | 141GB HBM3e | 4.8 TB/s | 700W | 4th gen | FP8 | NVLink-3 900 GB/s |
| B100 / B200 SXM | Blackwell | 192GB HBM3e | 8 TB/s | ~1000W | 5th gen | MXFP8 / MXFP6 / **NVFP4** (TE 2nd gen) | NVLink-4/7 1.8 TB/s |
| GB200 NVL72 | Blackwell+Grace | 72×192GB = 13.8TB HBM3e/机柜 | 8 TB/s/GPU | ~1200W/GPU | 5th gen | NVFP4 (20 PFLOPS FP4/superchip) | NVLink Switch |
| GB300 NVL72 (Blackwell Ultra) | Blackwell Ultra+Grace | 72×(高容量 HBM3e, 业界报道 ~288GB) | 高于 GB200 | ~1400W (液冷) | 5th gen+ | NVFP4 / MXFP8 | NVLink + NVSwitch |

来源：
- Hopper/Blackwell 架构与 B200 192GB、双 GB100 die (NV-HBI 10 TB/s)、5th gen Tensor Core、MXFP4/MXFP6、2nd gen Transformer Engine：https://en.wikipedia.org/wiki/Blackwell_(microarchitecture)
- GB300 NVL72 官方：72 Blackwell Ultra GPU + 36 Grace CPU，全液冷机柜级，**1.5× dense FP4 FLOPS、2× attention 性能（相对 Blackwell）**，10× 用户响应(TPS/user)、5× 能效(TPS/MW)、50× AI 工厂产出（相对 Hopper）；视频生成 5s 序列 = 4M tokens，Hopper 上约 90s，Blackwell Ultra 较 Hopper 30×：https://www.nvidia.com/en-us/data-center/gb300-nvl72/
- H200 141GB（Open-Sora 2.0 实际使用并确认）：arXiv 2503.09642 §6

### 集群与 AI 工厂视角
- **拓扑与网络**：fat-tree，GB300 NVL72 配 **Quantum-X800 InfiniBand 或 Spectrum-X Ethernet** + **ConnectX-8 SuperNIC**（800Gbps 级），跨机柜 400/800Gbps。Hopper 时代主流 400Gbps IB/Ethernet。
- **机柜级供电/散热**：GB300 NVL72 为**全液冷**机柜级系统；NVIDIA **DSX 平台**定位"最低 token/MW 成本"的 AI 工厂；GB300 较 Hopper 5× TPS/MW，能效跃升直接降低训练/推理每 token 能耗。
- **典型集群规模**：Open-Sora 2.0 用 192–224 张 H200；Megatron-LM 基准在 **6144 张 H100** 上训练 462B LLM（参考规模，非视频）；工业级视频预训练（Wan 14B / HunyuanVideo 13B）未公开，量级估计为数千卡·数周。

---

## 3. 训练框架

| 框架 | 定位 | 视频相关用法 | 来源 |
|---|---|---|---|
| **Megatron-LM / Megatron Core** | GPU 优化大规模训练库 | TP/PP/DP/EP/CP + FP16/BF16/FP8/FP4；6144×H100 训 462B 达 47% MFU；Dynamic CP（变长序列 1.48×）；**Megatron Bridge** 做 HF↔Megatron checkpoint 互转；导出 TensorRT-LLM | https://github.com/NVIDIA/Megatron-LM |
| **NVIDIA NeMo** | 端到端训练框架 | 与 TransformerEngine 集成做 FP8 训练 | TE README 集成列表 |
| **DeepSpeed** | 优化库 | ZeRO / ZeRO-Infinity / ZeRO-Offload / ZeRO++ / **Ulysses SP** / Ulysses-Offload / Arctic Long Sequence Training / ZenFlow / DeepNVMe / SuperOffload | https://github.com/microsoft/DeepSpeed |
| **ColossalAI** | 大模型并行系统 | Open-Sora 2.0、Allegro 的训练后端；tensor/sequence parallelism | https://github.com/hpcaitech/Open-Sora (§6) |
| **PyTorch FSDP** | 原生 | Wan 系列多卡推理/训练（`--dit_fsdp --t5_fsdp`） | Wan2.2 README |
| **视频/扩散专用训练代码库** | — | — | — |
| — CogVideoX-Fun | CogVideoX 改造管线 | 灵活分辨率多启动方式 | CogVideo README 友链 |
| — cogvideox-factory | 低成本微调 | **单卡 4090 即可微调 CogVideoX-5B** | 同上 |
| — DiffSynth-Studio | 扩散引擎 | Wan 2.2 全量训练/LoRA/FP8 量化/序列并行 | Wan2.2 README 友链 |
| — SAT | CogVideoX 权重训练 | 快速堆叠开发 | CogVideo README |
| — Allegro train.py | accelerate 脚本 | bf16 + `--enable_stable_fp32` + `--gradient_checkpointing` | Allegro README |
| — Open-Sora | ColossalAI | ZeRO-2 + CP + selective AC | arXiv 2503.09642 §6 |

**LoRA / 全量微调**
- Mochi：单卡 H100/A100 80GB 即可 LoRA（safetensors 格式）。来源：https://github.com/genmoai/mochi
- CogVideoX：diffusers 版 LoRA 显存更低；cogvideox-factory 单 4090 微调 5B。来源：CogVideo README
- Wan 2.2：DiffSynth-Studio 支持 LoRA 与全量训练。来源：Wan2.2 README

---

## 4. 视频 DiT 专属分布式策略（长序列 3D 注意力是瓶颈）

### 为什么视频比图像更吃 CP/SP
视频 token 数 = `T × H × W / (VAE 压缩比 × patch²)`，**多了时间维 T**，序列长度随帧数线性增长，而注意力复杂度 **O(N²)**：
- HunyuanVideo 720p/129f 经 4×8×8 VAE + patch 2 → 单视频 **~115K tokens**（Open-Sora 2.0 §3.1 实测）。
- Open-Sora 2.0 实测：**768p/129f 训练比 256p 慢 40×**，正是自注意力随 token 二次膨胀（§4.1.3）。
- GB300 官方举例：5s 视频生成序列需处理 **4M tokens**。
- 对比：单帧图像 token 数仅为视频的 1/T。因此视频 DiT 的扩展瓶颈在**序列维**而非参数维——CP/SP 是刚需，而非可选优化。

### 核心策略

| 策略 | 机制 | 通信成本 | 适用 | 来源 |
|---|---|---|---|---|
| **Context Parallelism (CP)** | 序列沿 seq 维切分到多卡，每卡独立算注意力（Ring Attention 思路） | 中（KV 环传/重叠） | 长视频、高分辨率 | Ring Attention arXiv 2310.01889；Megatron Dynamic CP |
| **Ulysses SP** | 沿注意力 head 切分，all-to-all 重排 | `4/N × O(p·hs)·L`（随 N 下降） | head 数可被 N 整除 | DeepSpeed Ulysses |
| **Ring SP** | 序列环形切分，KV 沿环传递 | `2 × O(p·hs)·L`（可重叠） | 序列长可被 N 整除 | xDiT |
| **USP (Unified SP)** | Ulysses + Ring 统一 API | 两者最小 | 通用 DiT 并行 | xDiT arXiv 2405.07719 |
| **Ring-Attention-Zig** | 锯齿切分均衡负载 | — | 负载不均场景 | Ring Attention 系列 |
| **PipeFusion** | patch 级流水线并行（利用扩散输入时序冗余） | `2 × O(p·hs)`（最低） | 多卡低通信 | NeurIPS 2025, xDiT |
| **3D 并行 (DP+TP+PP+CP)** | 组合 | — | 超大规模 | Megatron-LM |

来源：xDiT README 通信成本表与策略说明 https://github.com/xdit-project/xDiT ；USP 论文 arXiv 2405.07719；Ring Attention arXiv 2310.01889。

### 工程实践（Open-Sora 2.0 §6，实测于 H200 141GB）
- **VAE 训练**：把 tensor parallelism 适配到卷积层（按 input/output channel 切权重），降显存并避免越界索引。
- **MMDiT 训练**：**ZeRO-2 + Context Parallelism (CP)**，视频/文本序列沿 seq 维切到各卡，每卡独立算注意力，缓解高分辨率二次复杂度。
- **MFU 实测**：Stage 1/2（256p）DP+ZeRO-2 = **38.19% MFU**；Stage 3（768p）ZeRO-2+CP=4 = **35.75% MFU**。CP 在 H200 141GB 上单独使用即达最优"显存-算力"权衡。
- **VAE 并行**：xDiT 提供 DistVAE / Parallel VAE 防止 VAE 解码 OOM（VAE 编码/解码本身是另一瓶颈，尤其高压缩 AE）。

> 注：Wan 2.2 多卡推理从 xDiT USP 切换到 **PyTorch FSDP + DeepSpeed Ulysses**，说明生产侧更倾向原生 + DeepSpeed 栈；xDIT 则仍是 HunyuanVideo / Wan2.1 / LTX-2 / CogVideoX / Mochi 的并行推理首选。

---

## 5. 显存与计算优化

| 技术 | 作用 | 视频训练实例 | 来源 |
|---|---|---|---|
| **激活检查点 / 选择性重计算** | 以算换显存，只重算大激活（注意力） | Open-Sora selective AC（H200 141GB 允许更激进）；Allegro `--gradient_checkpointing` | arXiv 2503.09642 §6/H；Allegro README |
| **ZeRO-2 / ZeRO-3** | 分片 optimizer/grad/param | Open-Sora 用 ZeRO-2；ZeRO-3 全分片 | DeepSpeed README |
| **CPU / NVMe offload** | 显存溢出到 CPU/NVMe | DeepSpeed ZeRO-Offload / ZeRO-Infinity / **DeepNVMe** / ZenFlow(无停顿 offload) / SuperOffload；推理侧 Wan `--offload_model --t5_cpu`、Mochi `cpu_offload` | DeepSpeed README；Wan/Mochi README |
| **高压缩 VAE** | 降 token 数 → 降显存/算力 | Video DC-AE token 76K→19K（4×）；Wan2.2-VAE 4×32×32 | arXiv 2503.09642 §3.1/4.3 |
| **VAE tiling / slicing** | 分块解码防 OOM | CogVideoX `vae.enable_tiling/slicing`；Hunyuan/Open-Sora tiling | CogVideo README |
| **torch.compile / Triton kernel** | 单卡算子加速 | Open-Sora 用 PyTorch compile + Triton | arXiv 2503.09642 §6 |
| **注意力重计算策略** | 仅重算 attention，保留 FFN | selective AC 的核心 | Megatron Core / xDiT |

---

## 6. 混合精度训练（bf16 / fp8 / NVFP4）

### 训练精度现状
- **bf16 为主流训练精度**：Wan 2.1/2.2、HunyuanVideo、CogVideoX-5B、Allegro（bf16 + `--enable_stable_fp32`）、Open-Sora 2.0 均以 bf16 训练。
- **CogVideoX-2B 用 FP16 训练**（5B 用 BF16），官方建议用训练精度推理。来源：CogVideo README。
- **fp8 训练（Hopper TransformerEngine）**：TE 在 Hopper/Ada/Blackwell 上提供 FP8；**TE 官方收敛验证：FP8/MXFP8 与 BF16 训练 loss 曲线无显著差异**，已在 LLaMA2-7B/70B、MPT-13B、MoE-16B、LLM-8B 等验证。来源：https://github.com/NVIDIA/TransformerEngine （Convergence 节）。
- **Blackwell 新格式**：TE 2nd gen 支持 **MXFP8、NVFP4**；NVIDIA 称 **"NVFP4 以 16-bit 精度训练、4-bit 速度/效率"**，Nemotron 3 即以 NVFP4 训练（TE News 2025-08/12）。
- **集成**：TE 与 DeepSpeed、Megatron-LM、NeMo、HF Accelerate 等集成，支持 MoE / TP / SP / CP 融合。

### fp8 训练对下游异构推理精度的影响（关键桥接）
- **xDiT 实测：纯 FP8 注意力会引入视觉伪影**，故提供 **hybrid attention**——首尾 N 步用高精度后端、中间步用低精度后端。来源：xDiT README（FP8 attention backends 节）。
- 含义：低精度（fp8/fp4）注意力是**精度悬崖边缘**，对量化敏感。这直接传导到端侧——
  - **bf16 训练的模型**保留量化余量，端侧再降到 fp8/int8/int4 更稳健；
  - **fp8/NVFP4 训练的权重**已逼近精度下限，端侧进一步 int4/fp8 量化易崩（伪影、物体畸变）；
  - 端侧弱 fp8/int4 支持时，需沿用"步级精度调度"（首尾高精度步）。

---

## 7. 训练成本 / 能耗 / 时间数据点

### 唯一公开的完整成本拆解：Open-Sora 2.0（11B，H200，ColossalAI）
来源：arXiv 2503.09642 Table 2（假设 H200 租金 $2/GPU·hour）

| 训练阶段 | 数据集 | CP | 迭代 | GPU 数 | GPU·day | 成本 |
|---|---|---|---|---|---|---|
| 256px T2V | 70M | 1 | 85k | 224×H200 | 2240 | $107.5k |
| 256px T/I2V | 10M | 1 | 13k | 192×H200 | 384 | $18.4k |
| 768px T/I2V | 5M | 4 | 13k | 192×H200 | 1536 | $73.7k |
| **合计** | — | — | — | — | **4160 GPU·day** | **$199.6k** |

- 折算 ≈ **99,840 H200·hour**（4160×24）。
- 较 MovieGen、Step-Video-T2V（据公开信息估算）**便宜 5–10×**。
- 高分辨率是成本大头：768p 训练比 256p 慢 40×（§4.1.3），故策略是"低分辨率学运动 + 少量高分辨率微调"。

### 其他数据点
- **Megatron-LM**：6144×H100 训 462B LLM 达 47% MFU（非视频，作集群规模参考）。来源：Megatron README。
- **GB300 能效**：较 Hopper 5× TPS/MW、50× AI 工厂产出 → 训练/推理每 token 能耗大幅下降。来源：GB300 NVL72 官网。
- **推理时延（体现算力规模感）**：Allegro 20min/单 H100 → 3min/8×H100；HunyuanVideo 720p/129f/50step 1904s(1GPU)→337s(8GPU, 5.64×)；CogVideoX-5B 5s 视频 ~550s/H100；Wan2.1-1.3B 480p 5s ~4min/RTX4090。来源：各模型 README。
- **Wan / HunyuanVideo 训练成本未公开**：Wan 14B 在"数十亿图像+视频"上训练（arXiv 2503.20314 摘要），HunyuanVideo 13B 配"高效训练基础设施"（arXiv 2412.03603），但均未披露 GPU·hour。量级估计：远高于 Open-Sora 2.0 的 10 万 H200·hour，推测为**数百万 GPU·hour 级、千万美元级**预训练。

---

## 8. 导出给推理的产物

| 维度 | 训练侧做法 | 对端侧的影响 | 来源 |
|---|---|---|---|
| **checkpoint 格式** | **safetensors** 已成事实标准 | 端侧加载安全、可 mmap | Mochi `dit.safetensors`；Wan/Hunyuan/CogVideoX 均用 |
| **DiT 精度** | bf16 为主；CogVideoX-2B=FP16；HunyuanVideo/LTX/Wan(DiffSynth) 另发 **FP8 权重** | bf16→端侧 fp8/int8 量化余量大；FP8 训练/导出权重已近下限 | 各 README；TE Convergence |
| **VAE 精度（单独处理）** | VAE 通常 **FP32 / TF32** 保持高精度 | 端侧 VAE 必须至少 fp16/bf16，**fp8 VAE → 可见伪影**（像素空间放大） | Allegro"VAE best in FP32/TF32"；CogVideoX VAE FP32 |
| **文本编码器** | T5-XXL(~5B, 20GB+) / MLLM；常 offload 到 CPU | 端侧需**剪枝/量化/蒸馏**；CogVideoX 限 224 token；Hunyuan 用 MLLM+token refiner | Wan `--t5_cpu`；Mochi 单 T5-XXL；CogVideo README |
| **MoE vs Dense** | Wan 2.2 MoE(27B/14B激活) + Dense TI2V-5B | 端侧优先选 dense 5B；MoE 需专家路由 + 条件执行 | Wan2.2 README |
| **高压缩 VAE 导出** | DC-AE / Wan2.2-VAE 4×32×32 | 端侧 token 少→更快，但重建质量略降 | arXiv 2503.09642 §4.3 |
| **Megatron↔HF 互转** | Megatron Bridge | 训练用 Megatron、端侧用 HF/Diffusers 顺畅转换 | Megatron README |

**训练精度选择如何传导到端侧**
- 训练精度越低（fp8/NVFP4），权重越接近精度悬崖 → 端侧再量化空间越小。
- VAE 是精度硬下限：训练/导出 VAE 必须高精度，端侧不可过度量化。
- 文本编码器是端侧显存/算力大头，训练时的选型（T5 vs MLLM vs CLIP）直接决定端侧剪枝/替换难度。

---

## 桥接结论：训练侧决策如何影响异构端侧推理鲁棒性（给下游综合用）

1. **VAE 是跨设备精度硬下限**。训练侧须把 VAE 保持在 FP32/TF32 并随 checkpoint 单独导出；Apple Silicon / 消费级 GPU / 边缘 NPU 上 VAE 至少跑 bf16，fp8 VAE 会产生可见伪影（像素空间误差被放大）。这是训推异构中最脆弱的一环。

2. **bf16 训练为端侧量化留出余量，fp8/NVFP4 训练会逼近悬崖**。TE 验证 fp8 训练对 LLM loss 无损，但 xDiT 实测"纯 FP8 注意力产生视觉伪影"需 hybrid attention。意味着：在 Apple Silicon(MPS)/消费级 GPU(int8/int4)/边缘 NPU(int8) 上，bf16 训练的权重可平稳降精度；fp8/NVFP4 训练的权重再量化易崩，端侧需沿用"首尾高精度步"的步级精度调度。

3. **MoE 架构改变端侧部署形态**。Wan 2.2 的 27B/14B-激活 MoE 在数据中心省算力，但端侧（尤其 NPU）需支持专家路由与条件执行，工程复杂度高；面向异构端侧应导出 **dense 变体（如 Wan 2.2-TI2V-5B）** 作为主推理目标，MoE 仅用于云端高质场景。

4. **高压缩 VAE 是训推一致性的双刃剑**。DC-AE/Wan2.2-VAE（4×32×32）在训练侧降 token 数 4×、端侧推理更快，但重建质量下降 + 扩散适配变难。端侧若硬件弱，高压缩 VAE 反而更友好（少 token）；但需接受质量折损，并保证 VAE 解码精度。

5. **文本编码器是端侧显存/算力主负担，训练选型即决定端侧可行度**。T5-XXL(~5B) 对边缘 NPU/消费级 GPU 几乎不可全量部署，必须剪枝/量化/蒸馏；HunyuanVideo 的 MLLM 路线与 CogVideoX 的 224-token 截断是两种端侧减负策略。训练侧若能产出门类更轻的文本编码器或蒸馏对齐特征，可显著降低异构端侧门槛。

---

## 附录：一手来源 URL 汇总

- Wan 2.2：https://github.com/Wan-Video/Wan2.2
- Wan 2.1：https://huggingface.co/Wan-AI/Wan2.1-T2V-14B ；https://github.com/Wan-Video/Wan2.1
- Wan 技术报告：https://arxiv.org/abs/2503.20314
- HunyuanVideo：https://huggingface.co/tencent/HunyuanVideo ；https://arxiv.org/abs/2412.03603
- CogVideoX：https://github.com/THUDM/CogVideo
- LTX-Video：https://huggingface.co/Lightricks/LTX-Video
- Mochi 1：https://github.com/genmoai/mochi
- Allegro：https://github.com/rhymes-ai/Allegro
- Open-Sora 2.0：https://github.com/hpcaitech/Open-Sora ；https://arxiv.org/abs/2503.09642 ；HTML 全文 https://arxiv.org/html/2503.09642v3
- xDiT (USP/PipeFusion)：https://github.com/xdit-project/xDiT ；USP arXiv 2405.07719
- Megatron-LM：https://github.com/NVIDIA/Megatron-LM
- DeepSpeed：https://github.com/microsoft/DeepSpeed
- TransformerEngine：https://github.com/NVIDIA/TransformerEngine
- GB300 NVL72：https://www.nvidia.com/en-us/data-center/gb300-nvl72/
- Blackwell 架构：https://en.wikipedia.org/wiki/Blackwell_(microarchitecture)
