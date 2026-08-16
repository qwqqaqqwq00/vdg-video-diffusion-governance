# VDG 答辩 PPT 大纲（中文）

> 平台：VDG — Video Diffusion Governance（视频扩散治理平台）
> 主模型：LTX-2.3（Lightricks 视频 DiT，19B 参数，3D VAE 8×32×32）
> 定位：面向「NVIDIA 数据中心训练 → Apple Silicon / 消费级 NVIDIA / 边缘 NPU 推理」训推异构场景的工业级性能–能耗建模与 AI 治理平台
> 全部数据取自交付文档（README / INTEGRATION_REPORT / SYSTEM_DESIGN / SYSTEM_USE_CASES / ARCHITECTURE / TEST_REPORT / results.json / 顶层调研报告），未编造。

---

## 第 1 页：封面

**标题**：VDG — 视频扩散治理平台
**副标题**：训推异构场景下的性能–能耗建模与 AI 治理闭环
**定位一句话**：一条命令完成「诊断 MPS 数值发散 → 推荐修复补丁 → 规则禁用不适用加速 → Pareto 选最优 → 输出可执行 ComfyUI workflow JSON + patch 脚本」

要点：
- 主模型 LTX-2.3（用户实战修复过的模型），内置 Wan 2.1/2.2、HunyuanVideo、CogVideoX、Open-Sora 2.0 作跨模型参照
- 建在 Roofline 模型 + ComfyUI 式插件注册表脚手架上
- 杀手级能力：M4 Max + LTX-2.3 一条命令跑通完整治理闭环（68.85s → 28.57s，2.4×）
- 交付规模：17 设备 / 9 负载 / 19 技能 / 5 运行时发射器 / 8 标定锚点 / 500 pytest 全绿

建议配图：平台 Logo + 一张「训练侧 NVIDIA DC → 三类端侧推理」的异构图，箭头标注 checkpoint 流转与精度悬崖。

---

## 第 2 页：目录与评分维度对齐

要点：
- 答辩评分四维度对齐：业务价值(20%) / 作业能力(30%) / 抽象能力(30%) / 创新能力(20%)
- 交付件五件套：系统设计 / 可运行代码 / 系统用例 / 架构设计 / 测试报告（均有对应文档 + 代码）
- 演讲主线：痛点（P3-4）→ 方案（P5-9）→ 实证（P10-18）→ 结论（P19-23）
- 关键交付映射：SYSTEM_DESIGN / SYSTEM_USE_CASES / ARCHITECTURE / TEST_REPORT + src/vdg + tests/

建议配图：一张「四维度 × 五交付件」的矩阵表，每个格子标注对应页码与文档文件名。

---

## 第 3 页：问题背景——训推异构痛点

**核心问题**：开源视频生成已收敛到 DiT + Flow Matching + 3D VAE 范式，训练在 NVIDIA DC 以 bf16/fp8/NVFP4 完成，checkpoint 需部署到算力/精度/架构差异极大的端侧。

要点：
- 训练侧 vs 端侧鸿沟：算力精度（bf16 主 → MPS bf16/fp16、消费 NV fp8/fp4/int8、NPU int8/int4）
- 注意力后端鸿沟：训练 FA-3（H100 740 TFLOPS）→ MPS 无融合内核（math 回退）、消费 NV SageAttention、NPU 需原生移植
- 显存/带宽鸿沟：HBM3e 80–192GB / 3.35–8 TB/s → MPS 统一内存 64–512GB 但仅 410–819 GB/s；消费 NV 24–32GB；NPU 273 GB/s
- 本质矛盾不是「算力不够」而是「精度悬崖」：同一模型被推向精度下限时的隐蔽崩溃

建议配图：三列对比表（训练侧 NVIDIA DC / Apple Silicon / 消费 NV / 边缘 NPU），行=算力精度/注意力后端/显存带宽/数值稳定性，鸿沟列用红色标注。

---

## 第 4 页：精度悬崖——跨设备数值发散根因

**杀手级洞察**：六个崩溃点不是独立的，而是一条完整崩溃链——时间步嵌入投影误差 → AdaLN scale ≈ -1 → 灾难性抵消 → 整层清零 → 累积黑帧。

要点：
- ★最隐蔽：AdaLN 调制 `(1+scale)` 灾难性抵消——bf16 7 位尾数，`scale≈-1` 时 `(1+scale)→0` 丢全部有效数字（阈值 `|1+scale|<2^-7≈0.0078`）
- GELU-tanh 融合内核缺陷：MPS bf16 `|x|≥15` NaN（非溢出，是 Metal 融合内核 bug），fp16 `|x|>40` 溢出
- RMSNorm/LayerNorm 平方溢出、Softmax 大序列 `x-max` 下溢、VAE 大数值范围 GroupNorm 不稳
- 真实例证：用户 `MPS_BLACK_VIDEO_FIX.md` 切断崩溃链最关键两环（GELU + AdaLN）；2026 年仍在复现（ComfyUI #15315：M4 Max 全黑+音频 NaN）
- T2V 比 I2V 更易崩溃：文本条件经 timestep + text 双重调制，scale 动态范围更大

建议配图：崩溃链流程图（timestep 投影误差 → scale≈-1 → (1+scale)→0 → 层清零 → 黑帧），标注用户修复切断的两环。

---

## 第 5 页：解决方案总览——VDG 治理闭环架构

**一句话**：VDG 把「端侧部署决策」建模为「Roofline 性能 + 能耗 + AI 治理」三层闭环，输出可落地的决策、Pareto 备选与补丁指令。

要点：
- 治理五步闭环：诊断 → 规则 → 选择 → 修复 → 仿真（→ 报告），四代理共享 AgentContext
- 四层架构：插件层（自注册）→ 核心层（frozen）→ 治理层（闭环）→ 接入层（CLI + runtime 发射可执行产物）
- 核心不变量：蒸馏步数下降只由技能 `speedup` 建模，绝不与 config 降步数叠加（防双重计数）
- 闭环不止于仿真：治理决策发射为 ComfyUI JSON / diffusers 脚本 / torch 补丁 / LightX2V / MLX 命令
- 锚点标定：8 个真实测量锚点修正 roofline 系统偏差，把仿真校准到实测

建议配图：一张完整架构图——插件层（devices/loads/skills）→ 核心层（roofline/energy/simulator/calibration/registry）→ 治理层（diagnose→rules→select→repair→simulate）→ 接入层（CLI + runtime 5 发射器），右侧标出 GovernanceReport 输出。

---

## 第 6 页：核心抽象——Roofline 性能模型

**抽象精髓**：把「设备差异」压缩成 `(peak_flops[precision], mem_bw)` 两元组，同一模型在不同设备上的预测只换两个数。

要点：
- 每个去噪步拆成 attention / FFN 两段：`achievable = min(peak_flops, arithmetic_intensity × mem_bw)`，再除 FLOPs 得时间
- 注意力后端映射精度峰值：sage3→FP4、sage2→FP8、sage1→INT8、math→0.5×峰值；FFN 用步数精度峰值
- VAE 解码与文本编码各跑独立 roofline（VAE 是画质硬下限，需单独高精度建模）
- 能耗层：可插拔能耗模型——`TDPEnergyModel`（idle+(tdp-idle)×util 线性外推）+ `MeasuredEnergyModel`（pynvml 实时功率积分，无 NVIDIA 硬件透明回退 TDP）
- 核心结论印证：MPS 不是算力瓶颈而是带宽瓶颈——M4 Max 68.85s vs RTX 5090 8.87s（7.8×），远超算力比

建议配图：经典 Roofline 双对数曲线（算力 vs 算术强度），标注 MPS（memory-bound 段）与 5090（compute-bound 段）两个工作点。

---

## 第 7 页：核心抽象——插件注册表与四可插拔维度

**脚手架**：复刻 ComfyUI/pytest 式装饰器注册表（`core/registry.py`），`(kind, name) -> class` 扁平存储，导入 `vdg` 即自动发现，无需改中央代码。

要点：
- 四可插拔维度正交：设备(Device, 17) / 负载(Load, 9) / 技能(Skill, 19) / 能耗模型(EnergyModel, 2)
- 关键设计：基类已用 roofline 实现 `tokens_for`/`per_step_flops`/`memory_footprint`，子类**只填数据**不重写算法——「加新模型」退化成「填一张架构参数表」
- 装饰器双用例：`@register_device`（名字取类名）与 `@register_device("rtx_5090")`（显式命名）
- 稳健性：导入失败吞没（缺可选插件静默跳过，不毒化注册表）；技能组合次乘性数学 `product(speedup)^0.85`（指数<1 建模瓶颈转移与收益递减）
- 演进性：新增 EnergyModel/DeviceProfile 子类即可对接 DSX OpenUSD 数字孪生，无需改核心

建议配图：四维度表格（维度/基类/装饰器/实例数/扩展动作），右侧一段「新增设备」的 `@register_device` 代码片段（DeviceSpec dataclass）。

---

## 第 8 页：四层架构与治理四代理闭环

**治理层**：四代理串成闭环，用 `GovernanceDecision` 结构化输出「用哪个技能、什么配置、为什么」，而非黑盒推荐。

要点：
- `DiagnosticAgent`：数值探针（GELU/AdaLN/RMSNorm/Softmax 四 op 边界输入对比 cpu-fp32）→ 修复建议
- `RuleEngine`：四条硬规则（详见 P9）→ disabled/preferred/overrides
- `AccelSelectorAgent`：枚举单技能变体 + 对/三元组 + 5 命名配方（RTX5090 上 46 组合，M4 Max 因 R1 降至 24），逐一仿真按 (latency, energy, quality) Pareto 排序
- `RepairAgent`：patched_config + 落地补丁指令（LTX 三处 cast 模板）
- `SimulatorAgent`：最终权威仿真 + Policy.enforce 验收（能耗/延迟/质量/显存四约束）

建议配图：simulate→select→apply 闭环流程图（6 步：baseline→diagnose→rules→select→repair→simulate→report），每步标注产出物。

---

## 第 9 页：规则引擎——四条硬规则

**规则的价值**：把领域知识（调研报告 + 用户实战）固化成可执行硬规则，使治理决策可解释、可审计、可复现。

要点：
- **R1**：Apple Silicon 禁用 CUDA-only 注意力（sage/flash/compile），并**偏好 teacache**（MPS 上唯一设备无关的注意力侧加速）
- **R2**：能耗超预算 → `preferred_skills` 加入 `step_distill`（跨设备最大能耗杠杆，5.54×）
- **R3**：质量目标 >85 → 限制量化只能 `gguf_q4`（禁 nvfp4/int8）
- **R4**：int8-only 设备（Ampere/Jetson/Ascend）部署 fp8 训练权重 → 加 boundary-block bf16 保护（首 2 + 末 3 块）
- 规则与 Pareto 协同：规则筛掉不适用技能后，在受限空间内做真实可用的 Pareto 选择（如 M4 Max 不强行套负收益的 SageAttention）

建议配图：四规则卡片（R1-R4），每张标注触发条件 / 动作 / 对应调研出处；右侧 M4 Max 受限技能空间示意（禁用 sage/flash/compile，仅留 teacache/distill/vae_tiling）。

---

## 第 10 页：杀手级 Demo——M4 Max + LTX-2.3 一条命令

**命令**：`python -m vdg govern --device M4_Max --model LTX_2_3 --scenario ltx_t2v_480p`

一条命令完成四件事，全部数据落地、可复现：

要点：
- **(a) 检测 MPS 低精度**：仿真探针报 2/4 敏感 op 发散——`gelu_tanh=nan`（`|x|≥15`）、`adln_modulate=divergence`（`(1+scale)` 灾难性抵消，-0.999→-1.0）
- **(b) 推荐修复技能**：`gelu_fp32` + `adaln_fp32`——即 `MPS_BLACK_VIDEO_FIX.md` 三处 cast fp32，输出 grep 定位 + 验证补丁指令
- **(c) 选择加速**：R1 禁 sage/sliding_tile（CUDA only），偏好 teacache；Pareto 选 top=`teacache:thr0.2`（蒸馏质量不达标被排除）
- **(d) 输出 SimulationResult**：baseline 68.85s/24959J → **final 28.57s/11525J（2.4×）/ q87.73 / feasible**，含 24 个排名备选 + 2 段补丁指令
- 工程级诚实：`--real-probe` 切真机实测，本机 torch 2.12 已修复 GELU 内核 → 实测只报 AdaLN（仍选 teacache、仍 feasible），区分「真机当前状态」与「文档化风险阈值」

建议配图：终端输出截图（Governance Report），标注四项验收全中；或一张四步闭环图配最终数字。

---

## 第 11 页：真实修复——GELU/AdaLN fp32 cast 补丁

**来源**：用户实战 `MPS_BLACK_VIDEO_FIX.md`，已被提炼为设备无关的数值鲁棒补丁模板，并在 `skills/repair/` 与 `runtime/torch_runtime.py` 落地为真实补丁。

要点：
- 三处 cast 站点：GELU / AdaLN-attn（scale_msa/shift_msa/gate_msa）/ AdaLN-MLP（scale_mlp/...）
- 补丁模板原则：低精度后端上，敏感 op 的**中间计算强制 fp32**，仅边界张量按后端精度 cast
- `TorchRuntime.apply_all()`：按名称/类名启发式定位 `transformer_blocks.*` 的 GELU/AdaLN 站点，就地打真实补丁，支持 `unpatch()` 回滚
- 三粒度保精度可叠加：op 级（敏感 op fp32）/ block 级（首尾 boundary block bf16）/ 层集级（高脆弱层集 bf16 保护）
- 修复成本被显式权衡：`adaln_fp32` speedup 0.900、quality_delta +2.5、energy_ratio 1.06——修复纳入 Pareto 而非事后加补丁破坏加速

建议配图：补丁代码片段（`adln_modulate` fp32 cast：`x.float()` 后 `(1.0+s32)` 再 `.to(x.dtype)`）；右侧三粒度叠加示意图。

---

## 第 12 页：加速技能矩阵——19 技能 + 5 命名配方

**技能全景**：13 加速 + 6 修复（+ 数值探针），倍率全部取自一手调研报告并在 docstring 标注来源 URL。

要点：
- 加速技能（13）：step_distill / teacache / sage_attention / quantization / offload / vae_tiling / compile_graph / flash_attention / sliding_tile_attention / linear_attention / mlx_sdpa / context_window / diffusion_forcing
- 修复技能（6）：adaln_fp32 / gelu_fp32 / rmsnorm_fp32 / softmax_fp32 / vae_fp32 / boundary_block_bf16
- 5 命名配方：R1 distill_cache_sage_vae（30–60×）/ R2 sta_fp4_compile（5–7×）/ R3 gguf_offload_tile_cache（~2×，核心价值是 fit）/ R4 linear_nvfp4_compile（~38×）/ R5 trt_compile_graphs_vae（~3.5×）
- 技能组合数学：`SkillImpact` 四元组（speedup/memory_ratio/quality_delta/energy_ratio）次乘性组合
- 诚实标记：未注册技能降级为接地估计候选（`estimate_only`，Pareto 置后），Phase 2 已注册 7 技能，R2/R4 不再退化

建议配图：技能矩阵表（技能名 / 类别 / 加速倍率 / 质量代价 / 适用设备 / 来源），5 配方用色块标注组合。

---

## 第 13 页：单技能 Pareto 形状（RTX 5090 / LTX-2.3 / 480p·81f 锚点）

**洞察**：Pareto 前沿随约束移动——要画质选 v2，要极致速度选 distill，要省能+平衡选 v3，要塞进显存选 gguf/tiling/offload。

要点（`results.json` 真实数字）：
| 组合 | 延迟 | 能耗 | 画质 | 加速 | Pareto 定位 |
| baseline | 8.87s | 3949J | 84.0 | 1.00× | 参考点 |
| step_distill:4step | 1.60s | 712J | 81.0 | 5.54× | 最快/最省能，画质掉 3 |
| sage_attention:v3 | 2.26s | 704J | 83.5 | 3.93× | 省能+高画质平衡点 |
| sage_attention:v2 | 3.49s | 1397J | 83.9 | 2.54× | 画质损耗最小 |
| teacache:thr0.1 | 4.54s | 2020J | 83.7 | 1.96× | 设备无关、训练免费 |
| quantization:nvfp4 | 3.49s | 1086J | 82.0 | 2.54× | 省能，画质掉 2 |
| vae_tiling / offload | 9.71s / 13.70s | — | — | 0.91× / 0.65× | 为内存不为快 |

建议配图：三目标散点图（latency-x / energy-y / quality-气泡大小），Pareto 前沿用线连接，baseline 标红。

---

## 第 14 页：6 场景测试矩阵

**全 feasible**：6 场景覆盖四类设备族群，均跑完整 GovernancePipeline，治理层在 SLO/能耗预算/质量地板下自动选最优 policy-feasible 组合。

要点（`results.json` / TEST_REPORT 真实数字）：
| # | 场景 | baseline | final | 加速 | 能耗节省 | top 组合 |
| 1 | H100 + LTX-2.3 + 720p | 14.56s/7901J | 4.74s/2569J | 3.1× | 67.5% | step_distill:8step |
| 2 | RTX 5090 + NVFP4 | 8.87s/3949J | 1.95s/828J | 4.5× | 79.0% | teacache+sage_attention |
| 3 | RTX 4090 + distill | 11.27s/3930J | 2.48s/825J | 4.5× | 79.0% | teacache+sage_attention |
| 4 | M4 Max + LTX-2.3（killer） | 68.85s/24959J | 28.57s/11525J | 2.4× | 53.8% | teacache:thr0.2 |
| 5 | Jetson Thor + distill | 14.40s/1440J | 5.12s/543J | 2.8× | 62.3% | step_distill:8step |
| 6 | Ascend 910B + INT8 | 11.62s/2789J | 4.32s/1153J | 2.7× | 58.7% | step_distill:8step + boundary_block |

- 修复建议随设备分化：M4 Max=gelu+adaln；Ascend=int8 加 boundary_block_bf16；其余=adaln_fp32

建议配图：场景矩阵表（如上），加速比与能耗节省用条件色阶；右侧四类设备图标对应场景。

---

## 第 15 页：能耗量化对比与能耗约束

**核心公式**：能耗 = 功耗 × 延迟。这解释了为何 M4 Max 系统功耗与 5090 相当，单 clip 能耗却是 5090 的 6.3×。

要点：
- M4 Max：480W × 68.85s（带宽拉长延迟）→ 24959J，是 RTX 5090（3949J）的 6.3×——带宽是 Apple Silicon 真正的能耗杠杆
- Jetson Thor：130W × 14.4s → 1440J，单 clip 能耗最低（低功耗而非高速度），边缘场景能耗约束天然宽松
- 步数蒸馏是跨设备通用能耗杠杆：5090 3949J→712J（step_distill 单技能 5.54×），延迟随步数近线性下降而功耗不变
- 能耗约束由 Scenario.energy_budget_j + Policy.energy_budget_j（CLI `--energy-budget` 覆盖）共同定义，Policy.enforce 逐候选检查违规
- 典型预算区间：消费 NV 5–50kJ；Apple Silicon 需蒸馏+缓存大幅压缩；NPU 单 clip 能耗最低

建议配图：双柱状图（baseline vs final 能耗，6 设备），叠加功耗×延迟分解；标注 M4 Max 带宽杠杆。

---

## 第 16 页：仿真校准——roofline 预测 vs 实测锚点

**解决「仿真非 bit-exact」**：8 个真实测量锚点修正 roofline 系统偏差，把系统性偏差从模型中消掉。

要点：
- 标定公式：`calibration_scale = measured_latency / predicted_latency`（在锚点工作点计算）
- **-92% 误差被校准修正的实例**：M4 Max / Wan2.1-1.3B 基座预测 258.35s，实测锚点 4500s（mlx-examples README），roofline 误差 -92.3% → scale 12.92 修正为 3338.66s
- H100 / HunyuanVideo-13B 基座低估 65.8% → scale 2.92 修正
- 锚点逐一溯源：M4 Max MLX Wan2.1、RTX 4090D/H100 Wan2.1-I2V-14B（LightX2V）、H100 HunyuanVideo-13B、TeaCache 4.41×（speedup 锚点）、VAE 32→8GB（memory 锚点）
- 匹配稳健性：仅 kind=latency 锚点参与标定；speedup/memory 锚点只进报告表绝不误标定单卡延迟；多卡测量（H100×8）永不匹配单卡仿真

建议配图：标定前后对比条形图（M4 Max -92.3% 误差消除、H100 -65.8% 消除）；右侧锚点表溯源 URL。

---

## 第 17 页：运行时对接——五类可执行产物

**闭环最后一公里**：治理决策不再止于仿真，`vdg runtime --runtime <target>` 直接发射可执行产物，torch/diffusers 惰性导入，纯仿真环境不受影响。

要点：
| 运行时 | 产物 | 消费方式 |
| comfyui | 8 节点 `/prompt` API 工作流 JSON + 节点说明 + 可执行 torch 补丁脚本 | POST 到 `127.0.0.1:8188/prompt` 或 Load (API Format) |
| torch | 进程内 `TorchRuntime.apply_all()` 补丁脚本 | 对任意 nn.Module 就地打 gelu/adaln/rmsnorm/softmax/vae fp32，`unpatch()` 回滚 |
| diffusers | LTX-Video 管道脚本 | 加载 HF 模型 → 修复走 TorchRuntime + 加速 API 映射 |
| lightx2v | `python -m lightx2v.infer` 启动命令 | 4 步蒸馏栈（NVFP4/SageAttention/TeaCache） |
| mlx | `mlx_video_generate` 启动命令 | Apple Silicon 统一内存推理 |

- `RuntimeEnvelope.validate()` 消费前 fail-fast 校验配置键（如 comfyui+teacache 必须带 `rel_l1_thresh`），配错的技能不静默产生坏工作流
- ComfyUI 工作流示例：`CheckpointLoaderSimple → CLIPTextEncode×2 / EmptyLTXVLatentVideo → TeaCache → KSampler → VAEDecode → VHS_VideoCombine`（rel_l1_thresh 取自治理决策）

建议配图：5 运行时产物卡片 + ComfyUI 8 节点工作流拓扑图。

---

## 第 18 页：真实硬件冒烟测试验证

**证明 runtime 不是发射器存根**：`scripts/smoke_torch_runtime.py` 在本机 torch 2.12 / MPS 与 CPU 双路径跑通真实补丁。

要点：
- 构造微型 toy DiT（nn.Linear 堆栈 + nn.GELU + LTX 风格 AdaLN 块(scale_shift_table/attn1/attn2/ff) + RMSNorm + Softmax，3D token 输入）
- `TorchRuntime.apply_all()` 打 4 类修复补丁，断言 forward 仍可执行且有限、`_vdg_patched` 标记齐全
- 验证点：补丁后 forward 输出有限；4 类标记齐全；重放幂等（re-apply 全跳过）；`unpatch()` 恢复全部原始 forward 且零残留
- 真机 MPS + CPU 双跑通过，`tests/test_smoke_torch_runtime.py` 锁回归（2 个 pytest 子进程）
- AdaLN 块接口忠实于补丁所针对的 LTX BasicTransformerBlock 调用面（timestep 形状 [B,1,6*dim] 对齐 6 行 scale_shift_table）

建议配图：冒烟测试终端输出截图（`SMOKE TEST PASSED` + applied/unpatch 日志）。

---

## 第 19 页：创新点

**四项创新**：把分散的工程动作统一进可治理、可审计、可复现的闭环。

要点：
- **创新 1：数值鲁棒性修复 + 推理加速统一进治理闭环**——修复（MPS_BLACK_VIDEO_FIX 三处 cast）与加速（TeaCache/Sage/蒸馏/量化）不再割裂，修复成本（speedup 0.9 / energy_ratio 1.06）被纳入 Pareto 显式权衡，而非事后加补丁破坏加速收益
- **创新 2：规则引擎把领域知识固化**——R1-R4 四条硬规则（调研报告 + 用户实战提炼），使决策可解释可审计；承认设备限制，在受限技能空间选真实可用加速
- **创新 3：Pareto 多目标治理而非单点加速**——(latency, energy, quality) 三目标支配排序 + policy 验收，5 命名配方 + 全量枚举，而非「套一个最快技能」
- **创新 4：锚点标定 + 运行时发射器闭环落地**——8 锚点把真实测量与 roofline 预测耦合（-92% 误差被修正），5 运行时把治理决策落到可执行产物（真实硬件冒烟验证 TorchRuntime 非存根）

建议配图：四创新卡片，每张标注「传统做法 vs VDG 做法」对比。

---

## 第 20 页：业务结论

**四条数据支撑的业务结论**（均由 `results.json` 支撑）：

要点：
- **结论 1**：步数蒸馏是跨设备最大能耗杠杆——`step_distill:4step` 在 RTX 5090 单技能即 8.87s→1.60s / 3949J→712J（5.54×），设备无关、模型侧；「训练侧一次性投入，推理侧持续省电」，R2 规则据此自动 prefer 蒸馏
- **结论 2**：Apple Silicon 瓶颈是带宽不是算力——M4 Max 68.85s vs 5090 8.87s（7.8×），而 FP8 算力 108 TFLOPS 并非差 7.8×，鸿沟来自 546 GB/s vs 1.79 TB/s；给 MPS 堆算力无用，堆带宽与降 token 数（高压缩 VAE/蒸馏）才有效
- **结论 3**：同一量化在消费侧与数据中心侧回报相反——5090 NVFP4 既快又省（2.54×/3.6× 能耗，win-win）；NPU INT8 需 boundary 保护且 TRT 拒 transformer 层（收益有限）；DC fp8 已是常态回归训练精度选择；导出分档不能一刀切
- **结论 4**：VAE 是画质硬下限，修复有成熟分层范式——fp8 VAE 即可见伪影（`vae_fp32` 强制高精度）；AdaLN `(1+scale)` 是最隐蔽崩溃源；数值鲁棒性是导出分档一等公民而非部署后补救

建议配图：四结论配数据柱状图（蒸馏能耗杠杆 / MPS 带宽差距 / 量化回报对比 / VAE 精度下限）。

---

## 第 21 页：交付件完整性 + 评分维度自评

要点：
- **交付件五件套全部 ✅ 完整**：系统设计(SYSTEM_DESIGN) / 可运行代码(core+tests 500 绿+scripts) / 系统用例(SYSTEM_USE_CASES UC1-UC6) / 架构设计(ARCHITECTURE+Pareto+5 配方+4 结论) / 测试报告(TEST_REPORT+results.json)
- **评分四维度自评**：
  - 业务价值(20%) 高：直击训推异构痛点，杀手演示用用户实战修复过的 LTX-2.3，输出可落地补丁指令非玩具
  - 作业能力(30%) 高：17 真实设备(一手溯源)+9 真实模型(HF config)+19 技能+4 代理+5 配方+6 运行时+8 锚点；500 测试含真实 MPS 冒烟；6 场景全 feasible
  - 抽象能力(30%) 高：ComfyUI 式注册表；四层冻结契约；次乘性组合数学防双重计数；双探针分离规划与诊断
  - 创新能力(20%) 高：修复+加速统一进治理闭环；四硬规则；Pareto 多目标；锚点标定耦合真实测量
- 规模：62 个 .py 模块 / ~11558 LOC(src) / 20 测试模块 / compileall 干净

建议配图：交付件映射表（要求→产物→完成度✅）+ 评分雷达图（四维度均高）。

---

## 第 22 页：路线图与已知限制

**~96% 完成**：核心仿真 + 治理闭环 + 运行时对接 + 锚点标定 + 文档测试均已落地干净运行；真实硬件冒烟证明 TorchRuntime 非存根。

要点（诚实标记的已知限制）：
- 仿真为工程级近似：roofline + TDP，非 bit-exact；锚点只覆盖 8 个已发布测量点，无锚点对仍为工程估计
- 冒烟测试是 toy DiT 而非完整 2B 权重端到端推理（需下载权重 + ComfyUI/LightX2V 环境）
- Apple Silicon TFLOPS 为保守估计（Apple 不公开张量峰值，带宽才是真瓶颈）
- VDG 不自带 ComfyUI 自定义节点包——TeaCache/VAE 分块依赖 ComfyUI-TeaCache / VideoHelperSuite / ComfyUI-GGUF 等既有 pack（产物中已注明）
- 路线图：扩充更多设备族实测锚点；toy DiT → 完整 2B 权重端到端验证；训练侧成本锚点接入；DSX OpenUSD 数字孪生全链路扩展；AdaLN scale 远离精度悬崖正则（可发表研究点）

建议配图：完成度环形图（96%）+ 路线图时间轴（短/中/长期）。

---

## 第 23 页：总结与致谢

要点：
- 一句话定位：VDG 把「视频 DiT 训推异构部署」从经验试错变成可运行、可复现、数据落地的结构化治理
- 三大诉求的回答：① 组合训推方案（bf16 训练→分档导出→蒸馏→端侧部署）② 修复跨设备问题（五类敏感 op 保 fp32 + op/block/层集三粒度）③ 加速推理（蒸馏+缓存+注意力+VAE+量化可组合配方）
- 交付承诺：可运行（500 测试 + 真机冒烟）、可复现（results.json + gen_results.py 重生成）、数据落地（一手溯源 + 锚点标定）
- 核心价值：数值鲁棒性是导出分档一等公民，而非部署后补救
- 致谢：调研四份子报告 + MPS_BLACK_VIDEO_FIX 实战经验 + ComfyUI/LightX2V/MLX 开源生态

建议配图：一张「训练侧 → 治理闭环 → 端侧可执行产物」的端到端全景图收尾。

---

## 演讲节奏建议

> 总时长按 20 分钟答辩设计（含问答缓冲），每页约 50–70 秒；核心实证页（P10/P14/P16）适当延长，过渡页压缩。

| 页 | 标题 | 建议时长 | 节奏说明 |
|---|---|---|---|
| 1 | 封面 | 0:30 | 一句话定位 + 杀手演示数字钩子 |
| 2 | 目录与评分对齐 | 0:30 | 快速过评分维度，建立评审预期 |
| 3 | 问题背景 | 1:00 | 铺垫训推异构痛点，强调鸿沟 |
| 4 | 精度悬崖 | 1:00 | 崩溃链是核心洞察，讲透 AdaLN (1+scale) |
| 5 | 解决方案总览 | 1:00 | 架构图讲四层 + 五步闭环 |
| 6 | Roofline 抽象 | 0:50 | 两元组压缩设备差异，MPS 带宽结论铺垫 |
| 7 | 插件注册表 | 0:50 | 四可插拔维度 + 只填数据不写算法 |
| 8 | 四代理闭环 | 0:50 | 闭环骨架，为 P10 demo 蓄势 |
| 9 | 规则引擎 | 0:50 | R1-R4 四规则，强调可解释 |
| 10 | 杀手级 Demo | 1:30 | **重头戏**：一条命令四件事，终端输出截图 |
| 11 | 真实修复补丁 | 1:00 | GELU/AdaLN cast 代码 + 三粒度 |
| 12 | 技能矩阵 | 0:50 | 19 技能 + 5 配方，倍率溯源 |
| 13 | Pareto 形状 | 0:50 | 单技能散点图，前沿随约束移动 |
| 14 | 6 场景矩阵 | 1:20 | **重头戏**：全 feasible 表格 + 设备分化 |
| 15 | 能耗对比 | 0:50 | 功耗×延迟公式，M4 Max 6.3× |
| 16 | 仿真校准 | 1:10 | **重头戏**：-92% 误差被修正实例 |
| 17 | 运行时对接 | 0:50 | 5 产物 + ComfyUI 工作流拓扑 |
| 18 | 硬件冒烟 | 0:50 | SMOKE TEST PASSED 截图，证非存根 |
| 19 | 创新点 | 1:00 | 四创新，传统 vs VDG 对比 |
| 20 | 业务结论 | 1:10 | 四结论配数据 |
| 21 | 自评 | 0:50 | 交付件 ✅ + 评分雷达 |
| 22 | 路线图 | 0:40 | 96% + 诚实限制 |
| 23 | 总结 | 0:40 | 三诉求回答 + 收尾全景图 |
| — | 合计 | ~18:30 | 留 ~1.5 分钟问答缓冲 |

**节奏提示**：
- 前 5 页（P1-5）是「问题→方案」叙事，控制在 4 分钟内，避免在背景上耗时过多。
- P10 / P14 / P16 是三处实证高潮，各给足 1.5 分钟，用真实数字与截图说话。
- P19-20（创新+结论）是评分加分页，把「为什么这是工业级治理而非玩具」讲透。
- 若答辩限时 15 分钟，优先压缩 P6/P7/P8/P13（抽象细节页），保留 P10/P14/P16/P19/P20。
