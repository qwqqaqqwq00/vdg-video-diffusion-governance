# VDG 架构设计

> 交付件对应：AI 负载建模子方向「架构设计：建模分析的过程如何指导系统设计/优化结果，给出基于建模的业务结论，重点阐述性能-能耗权衡策略」。
> 主模型：LTX-2.3。量化数据来自 `results.json`。

---

## 1. 架构总览

```
┌─────────────────────────────────── 插件层（自注册）──────────────────────────────────┐
│ devices/ (13)        loads/ (9)        skills/accel (7)   skills/repair (5)          │
│ DeviceProfile        LoadModel         Skill(kind=accel)  Skill(kind=repair)          │
│  @register_device     @register_load    @register_skill    @register_skill            │
└───────────┬────────────────┬──────────────────┬──────────────────┬───────────────────┘
            │                │                  │                  │
            ▼                ▼                  ▼                  ▼
┌─────────────────────────────────── 核心层（frozen）──────────────────────────────────┐
│ contracts.py   DeviceSpec/DeviceProfile/VideoDiTLoad/LoadModel/Skill/SkillImpact/      │
│                GovernanceDecision/DeviceCategory  (FROZEN 公共接口)                    │
│ roofline.py    token_count/per_step_flops/roofline/predict_step_time/vae_decode_flops │
│ energy_model.py  EnergyModel/TDP/Measured  (@register_energy_model)                    │
│ scenario.py    Scenario/ScenarioLibrary/BUILTIN_SCENARIOS                              │
│ simulator.py   PerformanceEnergySimulator + SimulationResult + AgentContext            │
│ registry.py    Registry/REGISTRY/discover()  (ComfyUI 式 (kind,name)->class)           │
└───────────┬──────────────────────────────────────────────────────────────┬────────────┘
            │                                                              │
            ▼                                                              ▼
┌──────────────────────────── 治理层（闭环）──────────────────────────────────────────┐
│ governance/                                                          │  输出
│  pipeline.py   GovernancePipeline.run()  ── resolve→diagnose→rules    │   ──► GovernanceReport
│  policy.py     Policy/Violation  (energy/SLO/quality/memory 验收)     │       (决策+违规+补丁指令
│  rules.py      RuleEngine R1-R4  (设备/质量/能耗/格式 硬规则)          │        +Pareto 备选)
│ agents/                                                              │
│  diagnostic.py  NumericalProbe → 修复建议                              │
│  accel_selector.py  枚举组合+仿真+Pareto 排序+5 命名配方               │
│  repair_agent.py    patched_config + 落地补丁指令                       │
│  simulator_agent.py 最终权威仿真 + policy 验收                         │
└──────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────── 接入层 ────────────────────────────────────┐
│ cli.py   vdg simulate | calibrate | govern | probe | runtime | report    │
│ runtime/ comfyui_emitter / torch_runtime / diffusers_runtime /           │
│          lightx2v_emitter / mlx_emitter   （治理决策 → 可执行产物）       │
│ core/calibration  CalibratedSimulator（真实测量锚点标定仿真）             │
└─────────────────────────────────────────────────────────────────────────┘
```

分层职责：**插件层**只填数据（spec/characteristics/predict），**核心层**frozen 提供 roofline+能耗+
组合数学+锚点标定，**治理层**把核心仿真串成"诊断→规则→选择→修复→仿真"闭环，**接入层**把治理结论翻译成 CLI
与真实运行时（`runtime/` 包发射 ComfyUI/diffusers/torch/LightX2V/MLX 可执行产物）。任何一层都可独立替换：
换插件不改核心，换能耗模型不改仿真器，换治理代理不改闭环骨架。

---

## 2. 建模分析如何指导优化：simulate → select → apply 闭环

治理管道 `GovernancePipeline.run` 用一个共享 `AgentContext` 把建模分析串成指导部署的闭环：

```
          ┌─────────── 1. baseline ───────────┐
          │  无技能基线仿真 → SimulationResult  │  (喂给 R2 能耗判断)
          └───────────────┬───────────────────┘
                          ▼
          ┌─────────── 2. diagnose ───────────┐
          │  NumericalProbe(4 op) → 发散?      │  → 修复建议(GovernanceDecision)
          │  has_failure → 推荐匹配 repair skill│
          └───────────────┬───────────────────┘
                          ▼
          ┌─────────── 3. rules ──────────────┐
          │  RuleEngine R1: Apple→禁 CUDA 注意力│  → disabled/preferred/overrides
          │              R2: 能耗超→prefer 蒸馏 │     + boundary-block 决策
          │              R3: 质量>85→限量化     │
          │              R4: int8-only→boundary │
          └───────────────┬───────────────────┘
                          ▼
          ┌─────────── 4. select ─────────────┐
          │  AccelSelectorAgent 枚举           │
          │   单技能变体 + 显式对/三元组        │  → ranked candidates
          │   + 5 命名配方                     │     + Pareto front
          │   逐一 simulate → (lat,ene,q) Pareto│     + policy 验收
          └───────────────┬───────────────────┘
                          ▼
          ┌─────────── 5. repair ─────────────┐
          │  RepairAgent 应用修复决策           │  → patched_config
          │  生成落地补丁指令(LTX 三处 cast)    │     + patch_instructions
          └───────────────┬───────────────────┘
                          ▼
          ┌─────────── 6. simulate ───────────┐
          │  SimulatorAgent 最终权威仿真        │  → final_result
          │  (selected accel + repair skills    │     + violations
          │   + patched config) + policy.enforce│
          └───────────────┬───────────────────┘
                          ▼
                   GovernanceReport
          (决策+基线/最终对比+备选+补丁指令+违规+probe)
```

**闭环的关键约束**：步骤 4 的 Pareto 排序与步骤 6 的最终权威仿真**用同一套 step 数**
（均为 `scenario.steps`），蒸馏的步数下降只由技能 `speedup`（相对 `baseline_steps` 的边际倍率）建模，
**绝不**在 config 里同时降 `steps` 又叠加 speedup——否则会双重计数。这是平台正确性的核心不变量。

---

## 3. 性能-能耗权衡：Pareto 前沿 + 5 命名配方

### 3.1 Pareto 三目标

`AccelSelectorAgent` 把每个技能组合仿真成 `(latency_s, energy_j, quality_score)` 三元组，
用经典支配关系排序：A 支配 B 当且仅当 A 在三目标上均不劣且至少一项严格更优。未被任何候选支配的组成
Pareto 前沿（`pareto_rank=0`），再按 `(feasible, preferred-hit, recipe, latency, energy)` 排序选 top。
`Policy.enforce` 在每个候选上检查能耗/延迟/质量/显存四约束，违规即标记 infeasible。

### 3.2 单技能 Pareto 形状（RTX 5090 / LTX-2.3 / 480p·81f，`results.json`）

| 组合 | 延迟 | 能耗 | 画质 | 加速 | Pareto 定位 |
|---|---|---|---|---|---|
| baseline | 8.87s | 3949J | 84.0 | 1.00× | 参考点 |
| step_distill:4step | 1.60s | 712J | 81.0 | 5.54× | **最快/最省能**，画质掉 3 |
| sage_attention:v3 | 2.26s | 704J | 83.5 | 3.93× | 省能+高画质平衡点 |
| sage_attention:v2 | 3.49s | 1397J | 83.9 | 2.54× | 画质损耗最小 |
| teacache:thr0.1 | 4.54s | 2020J | 83.7 | 1.96× | 设备无关、训练免费 |
| compile_graph:trt | 4.07s | 1631J | 83.5 | 2.18× | 无画质损(TRT-FP8 微损) |
| quantization:nvfp4 | 3.49s | 1086J | 82.0 | 2.54× | 省能，画质掉 2 |
| quantization:gguf_q4 | 8.18s | 3641J | 83.0 | 1.08× | **为 fit 不为快** |
| vae_tiling | 9.71s | 4319J | 83.9 | 0.91× | **为内存不为快** |
| offload:0.5 | 13.70s | 6096J | 84.0 | 0.65× | **以延迟换内存** |

Pareto 前沿会随约束移动：要画质 → v2；要极致速度 → distill；要省能+平衡 → v3；要塞进显存 → gguf/tiling/offload。

### 3.3 五个命名配方（合成报告 4.3，`accel_selector.py` RECIPES）

| 配方 | 技能组合 | 目标设备 | 接地加速 |
|---|---|---|---|
| R1 distill_cache_sage_vae | CoDMD 4 步 + TeaCache + SageAttention2 + VAE 分块 | 消费 NV | 30–60× |
| R2 sta_fp4_compile | STA + SageAttention3 + region compile + FP8 | Blackwell 5090 | 5–7× |
| R3 gguf_offload_tile_cache | GGUF Q4 + offload + VAE 分块 + TeaCache | 低显存 NV + MPS | ~2×(核心价值:fit) |
| R4 linear_nvfp4_compile | SANA-Video 线性注意力 + NVFP4 + compile | 5090/H100 | ~38× |
| R5 trt_compile_graphs_vae | TensorRT FP8 + compile + CUDA Graphs + INT8 VAE | 消费 NV + Jetson | ~3.5× |

配方引用技能在 Phase 2 前尚未全部注册，选择器对未注册者**降级为接地估计候选**（用报告加速倍率中点
缩放基线，标记 `estimate_only`，Pareto 排序置后），诚实区分"已仿真"与"估计"。Phase 2 已注册
STA/linear_attention 等 7 个技能，R2/R4 不再退化为 estimate_only。

---

## 4. 建模分析指导的优化结果

闭环不只给数字，还给**可落地的优化路径**（`results.json` governance_runs）：

| 设备/场景 | 基线 | 最终(治理后) | top 组合 | 修复 |
|---|---|---|---|---|
| RTX5090 / 480p·81f | 8.87s/3949J | 1.95s/828J/q86.1 | teacache+sage_attention | adaln_fp32 |
| RTX4090 / 480p·81f | 11.27s/3930J | 2.48s/825J/q86.1 | teacache+sage_attention | adaln_fp32 |
| M4_Max / 480p·81f | 68.85s/24959J | 28.57s/11525J/q87.7 | teacache:thr0.2 | gelu_fp32+adaln_fp32 |
| Thor T5000 / shortclip | 3.0s/299J | 1.2s/123J/q85.7 | teacache:thr0.2 | adaln_fp32 |
| RTX5090 / long_video | 148.9s/66257J | 31.7s/10462J/q84.5 | sage_attention:v3 | adaln_fp32 |

注意 M4_Max 的加速栈受限：R1 规则禁用了 CUDA-only 注意力技能（sage/flash/compile），Apple Silicon 上
可用的只有设备无关的 TeaCache/蒸馏/VAE 分块——治理偏好 TeaCache 后选 `teacache:thr0.2`（2.4×），
而不是强行套一个在 MPS 上无内核的负收益技能。这正是"建模指导优化"的体现——**承认设备限制，
在受限技能空间内选真实可用的加速**。

---

## 5. 业务结论

基于建模分析的三条核心业务结论（均由 `results.json` 数据支撑）：

### 结论 1 — 步数蒸馏是跨设备最大能耗杠杆

`step_distill:4step` 在 RTX 5090 上单技能即 8.87s→1.60s / 3949J→712J（5.54× 延迟、5.5× 能耗），
且 `applies_to` 为全类别（设备无关、模型侧）。这是"训练侧一次性投入（CoDMD ~A100 级），
推理侧持续省电"的范式——**对任何能耗受限场景，蒸馏是首选杠杆**，远超其他单技能。
治理管道 R2 规则正是据此在能耗超预算时自动 `preferred_skills.add(step_distill)`。

### 结论 2 — Apple Silicon 的瓶颈是带宽，不是算力

M4 Max 480p·81f 基线 68.85s vs RTX 5090 的 8.87s（7.8×），而 M4 Max 估算 FP8 算力 108 TFLOPS
并不比 5090 的 419 TFLOPS 差 7.8×——延迟鸿沟来自 **546 GB/s vs 1.79 TB/s 的带宽差**。
roofline 模型显示 MPS 的 attention 段在 `math` 后端（0.5×峰值）+ 低带宽下严重 memory-bound。
业务含义：**给 Apple Silicon 堆算力无用，堆带宽与降 token 数（高压缩 VAE/蒸馏）才有效**；
且 Apple Silicon 无融合注意力内核，SageAttention/FlashAttention 不可用（R1 禁用），加速栈天然受限。

### 结论 3 — 同一量化在消费侧与数据中心侧回报相反

- **消费 NV（5090）**：NVFP4 量化 8.87s→3.49s / 3949J→1086J（2.54× 延迟、3.6× 能耗），
  既快又省（FP4 tensor core `energy_ratio=0.7`），是 win-win。
- **边缘 NPU**：INT8 在 Ascend/Jetson 上需 boundary-block bf16 保护（R4），且 TensorRT INT8
  拒收 transformer 层（仅 0.9% 体积增益），有效加速有限；HiF8 需保首 2+末 3 块高精度。
- **数据中心**：fp8 训练已是常态，量化不再是推理加速手段而回归训练精度选择。

业务含义：**导出分档不能一刀切**——同一量化方法在不同设备族上"既快又省"与"需保护且收益有限"并存，
治理管道 R3/R4 规则据此按设备类别与质量目标自动调整量化允许集与 boundary 保护。

### 结论 4 — VAE 是画质硬下限，修复有成熟分层范式

跨设备鲁棒报告与 `NumericalProbe`（本机 M4 Max 实测 `adln_modulate=divergence`）一致表明：
VAE 解码不可与 DiT 同步量化（fp8 VAE 即可见伪影，`vae_fp32` 技能强制 VAE 高精度）；
AdaLN `(1+scale)` 灾难性抵消是最隐蔽崩溃源（`adaln_fp32` 三处 cast 即用户实战修复）。
业务含义：**数值鲁棒性不是部署后补救，而是导出分档的一等公民**——治理管道把诊断→修复→仿真串成闭环，
修复技能的 `predict()` 影响被纳入最终 Pareto，使"修复成本"（如 fp32 调制 ~10% 延迟、`energy_ratio=1.06`）
在加速选择时被显式权衡，而非事后加补丁破坏加速收益。

## 6. 仿真校准（锚点标定）

**解决 INTEGRATION_REPORT 已知限制 #1**：roofline + TDP 仿真为工程级近似，非 bit-exact。
`src/vdg/core/calibration.py` 引入**锚点标定**——凡存在同 (设备, 负载, 分辨率) 的真实测量点，就用实测
修正预测，把系统性偏差（如 Apple Silicon 带宽瓶颈、14B 注意力开销）从模型中消掉：

```
calibration_scale = measured_latency_s / predicted_latency_s   （在锚点自身工作点计算）
```

- **匹配**：设备+负载按归一化名称匹配（`M4_Max` ≡ `M4 Max`）；分辨率像素数相对误差 ≤20% 取最近锚点；
  仅 `kind="latency"` 锚点参与标定——speedup/memory 锚点（TeaCache 4.41x、VAE 分块 32→8GB）只进报告表，
  绝不误标定单设备延迟。多卡测量（`H100_x8`）同理永不匹配单卡仿真。
- **应用**：`CalibratedSimulator(PerformanceEnergySimulator)` 是透明 drop-in（isinstance 兼容），先跑基座
  仿真，再在锚点工作点（锚点自己的分辨率/帧数/步数）重跑一次取比例，乘回当前预测的 latency + energy +
  breakdown（TDP 下 energy ∝ latency）；锚点命中时把 `calibration: anchor <设备>/<负载> @ <分辨率> <帧>f
  <步> steps … scale <值>` 写入 `result.warnings` 溯源，无锚点则 scale=1.0 并告警 `no anchor, engineering
  estimate`。技能组合数学不受影响（标定只修系统偏差，技能仍次乘性叠加）。
- **接入**：`GovernancePipeline(simulator=CalibratedSimulator())` 注入即全链路标定（构造参数本已存在，
  代理链已透传）；CLI `vdg calibrate --device X --model Y --scenario S` 打印标定预测 vs 实测 + 相对误差表。

### 6.1 锚点表（全部为真实测量数据，逐一溯源）

| 设备 | 负载 | 分辨率/帧/步 | 实测延迟 | 来源 |
|---|---|---|---|---|
| M4 Max (MLX) | Wan2.1-T2V-1.3B | 832×480 · 81f · 50 步 | 4500 s（~90 s/it） | [mlx-examples Wan2.1 README](https://github.com/ml-explore/mlx-examples/blob/main/video/wan2.1/README.md) |
| RTX 4090D ×1 | Wan2.1-I2V-14B-480P | 832×480 · 81f · 40 步 | 810.4 s（20.26 s/it） | [LightX2V README](https://github.com/ModelTC/LightX2V) |
| H100 ×1 | Wan2.1-I2V-14B-480P | 832×480 · 81f · 40 步 | 207.2 s（5.18 s/it） | [LightX2V README](https://github.com/ModelTC/LightX2V) |
| H100 ×8 | Wan2.1-I2V-14B-480P | 832×480 · 81f · 40 步 | 30.0 s（0.75 s/it）* | [LightX2V README](https://github.com/ModelTC/LightX2V) |
| H100（80G 级）×1 | HunyuanVideo-13B | 1280×720 · 129f · 50 步 | 1904.08 s | [HunyuanVideo README](https://github.com/Tencent-Hunyuan/HunyuanVideo) |
| H100 ×8 | HunyuanVideo-13B | 1280×720 · 129f · 50 步 | 337.58 s * | [HunyuanVideo README](https://github.com/Tencent-Hunyuan/HunyuanVideo) |
| TeaCache 评测（GPU/分辨率未公开） | Open-Sora-Plan | — | 99.65 → 22.62 s（4.41×）† | [TeaCache](https://github.com/LiewFeng/TeaCache) |
| H100（假设，ComfyUI 基准未注明） | HunyuanVideo-13B | 1280×720 · 129f · VAE 解码 | 32 → 8 GB（4×）‡ | [ComfyUI HunyuanVideo 示例](https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_video/) |

\* 多卡测量：不参与单卡标定，仅作报告数据点。 † kind=speedup：技能级倍率锚点（4.41×，VBench −0.07%），非设备延迟标定。 ‡ kind=memory：VAE 时间分块峰值显存锚点，不标定延迟。

实测标定效果（`vdg calibrate`）：H100/HunyuanVideo-13B 基座低估 65.8% → 锚点工作点 scale 2.92 修正；
M4 Max/MLX 基座低估 92.3% → scale 12.9 修正（正是「Apple 带宽瓶颈」结论 #2 的量化体现）。

---

## 7. 运行时对接：从 envelope 到可执行产物

**解决 INTEGRATION_REPORT 已知限制 #2（Skill.apply 仅为 stub envelope）**：`src/vdg/runtime/` 把治理决策
变成**可直接执行的运行时产物**，`vdg runtime --runtime <target>` 在治理管道跑完后直接发射，闭环不再止于仿真：

```
GovernanceReport.decisions
   │  RuntimeEnvelope.from_dict → validate()   （必需配置键 fail-fast）
   ▼
┌─ runtime/ ─────────────────────────────────────────────────────────────┐
│ comfyui_emitter    build_workflow → /prompt API JSON + markdown 说明     │
│                    + render_patch_script（可执行 torch 补丁段）          │
│ torch_runtime      TorchRuntime：按名称/类名启发式定位敏感子模块         │
│                    （gelu/adaln/rmsnorm/softmax + LTX scale_shift_table  │
│                    AdaLN 标记），就地打 vdg.skills.repair 真实补丁，      │
│                    unpatch() 从 _vdg_original_forward 回滚               │
│ diffusers_runtime  LTX 管道加载（None-safe）→ 修复走 TorchRuntime        │
│                    + 加速 API 映射（enable_teacache / enable_tiling /    │
│                    enable_model_cpu_offload / torch.compile /            │
│                    set_num_inference_steps）                             │
│ lightx2v_emitter   python -m lightx2v.infer 启动命令                     │
│                    （--steps/--quant/--attn/--use_teacache/--fp32_ops…） │
│ mlx_emitter        mlx_video_generate 启动命令                           │
│                    （--quantize/--steps/--use_teacache/--tiling…）       │
└─────────────────────────────────────────────────────────────────────────┘
```

- **envelope → 可校验产物**：技能 `apply()` 返回的 envelope dict 经 `RuntimeEnvelope.from_dict` 升级为
  一等 dataclass（kind∈config|patch|workflow，target_runtime∈comfyui|diffusers|lightx2v|mlx|torch|tensorrt），
  `validate()` 按 (target_runtime, skill) 检查必需配置键（如 comfyui+teacache 必须带 `rel_l1_thresh`、
  comfyui+vae_tiling 必须带 tile_size/overlap/temporal_size/temporal_overlap），错误配置在消费前 fail-fast，
  而不是静默产生坏工作流；无运行时绑定的技能（如概念技能）标记为 advisory。
- **TorchRuntime 是 stub 的真实化**：repair 决策可对真实 nn.Module 就地生效——`find_sensitive_modules` 按
  名称/类名启发式定位任意 DiT 的 GELU/AdaLN/RMSNorm/Softmax 站点，`apply_all` 逐一打 `vdg.skills.repair`
  的真实 patch 函数（幂等跳过已打站点），`unpatch` 回滚原始 forward。所有 torch/diffusers 导入惰性化，
  纯仿真环境（无 torch）仍可 `import vdg` 并正常发射 comfyui/lightx2v/mlx 产物。
- **示例**：`vdg runtime --device RTX5090 --model LTX_2_3 --scenario ltx_t2v_480p_81f --runtime comfyui`
  发射 8 节点工作流 `CheckpointLoaderSimple → CLIPTextEncode×2 / EmptyLTXVLatentVideo → TeaCache →
  KSampler → VAEDecode → VHS_VideoCombine`（TeaCache 的 `rel_l1_thresh` 直接取自治理决策），可 POST 到
  `http://127.0.0.1:8188/prompt` 执行；`--runtime torch` 发射 `TorchRuntime.apply_all()` 补丁脚本。
  详见 SYSTEM_USE_CASES UC6 与 `vdg runtime --help`。
