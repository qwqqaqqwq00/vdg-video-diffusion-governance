# VDG 系统用例

> 交付件对应：AI 负载建模子方向「系统用例：设备的上下文、关键用例和逻辑视图，主要业务的系统用例，包含能耗约束条件」。
> 主模型：LTX-2.3。量化数据来自 `results.json`。

---

## 1. 设备上下文

VDG 覆盖四类端侧设备族群，每类的算力/带宽/精度/功耗上下文决定了它在该平台上的可行加速路径：

| 设备族群 | 代表设备 | 显存 | 带宽 | 功耗(TDP) | 支持精度 | 注意力后端 | 上下文特征 |
|---|---|---|---|---|---|---|---|
| 消费级 NVIDIA | RTX 4090 / 5090 / 6000 Ada | 24–48 GB | 0.96–1.79 TB/s | 300–575 W | bf16/fp16/fp8/(fp4) | flash/sdpa/sage2/(sage3) | 最成熟；ComfyUI 原生+LightX2V；5090 有 NVFP4 |
| Apple Silicon | M4 Max / M3 Ultra / M2 Ultra | 64–512 GB 统一 | 546–819 GB/s | 370–480 W(系统) | bf16/fp16/fp8 | mlx_sdpa/math | 统一内存装大模型，但**带宽是瓶颈**，无融合注意力内核 |
| 工业边缘 NPU | Jetson Thor T5000/T4000、Orin、Ascend 910B、MLU590、RK3588、Hexagon | 16–128 GB(多统一) | 51–1200 GB/s | 8–310 W | int8/(fp4/fp8/bf16) | vendor_attn/math | Thor 算力足但 273 GB/s 带宽拖累；低端移动 NPU 不可行 |
| 数据中心 NVIDIA | H100 / H200 / B200 / GB300 | 80–192 GB | 3.35–8 TB/s | 700–1200 W | bf16/fp8/(nvfp4) | flash/sdpa/sage2/(sage3) | 训练侧主战场；fp8/NVFP4 训练新趋势 |

> 能耗约束详见第 5 节"能耗约束条件"。

---

## 2. 逻辑视图（组件图）

```
                        ┌─────────────────────────────────────────────┐
                        │              VDG 治理闭环                    │
                        │                                             │
   设备/负载/技能 ──►   │  diagnose ─► rules ─► select ─► repair ─► simulate ─► report
   (注册表自注册)       │   │          │         │          │          │           │
                        │   │  AgentContext (device, load, scenario,    │           │
                        │   │  config, skills_applied, results[])       │           │
                        │   ▼          ▼         ▼          ▼          ▼           ▼
                        │ Numerical  Rule    AccelSelector Repair     Simulator   Governance
                        │ Probe      Engine  Agent         Agent      Agent        Report
                        │  (4 ops)   (R1-R4) (枚举+Pareto) (补丁+指令) (最终权威)  (决策+违规)
                        └─────────────────────────────────────────────┘
                               ▲                          │
                               │                          ▼
   ┌────────────┐    ┌──────────────────┐    ┌──────────────────────────┐
   │ core/       │    │ core/simulator   │    │ runtime/  (决策→产物)     │
   │ roofline    │◄──►│ PerformanceEnergy│───►│ RuntimeEnvelope 校验       │──► ComfyUI / diffusers
   │ energy_model│    │ Simulator        │    │ comfyui_emitter           │    / torch 补丁 /
   │ scenario    │    │  (roofline+能量  │    │ torch_runtime             │    LightX2V / MLX /
   │ registry    │    │   +技能组合)     │    │ diffusers_runtime         │    TensorRT
   │ contracts   │    └──────────────────┘    │ lightx2v/mlx emitters     │
   └────────────┘                             └──────────────────────────┘
```

四个治理代理共享 `AgentContext`（设备+负载+场景+已积累的仿真结果），逐步把"未技能化的基线"
加工成"已选技能组合 + 已修复数值 + 已验收"的最终结果。`runtime/` 包把治理结论发射为**可执行产物**
（ComfyUI 工作流 JSON / diffusers 脚本 / torch 补丁 / LightX2V / MLX 命令），使闭环不限于仿真而能
直接落地到用户的 ComfyUI/diffusers 工作流；`RuntimeEnvelope.validate()` 在消费前 fail-fast 校验配置键。

---

## 3. 主要业务用例

### UC1 — Profile + Simulate 部署评估

**触发**：拿到一个新模型/设备，想知道"能不能跑、跑多快、耗多少电"。
**主流程**：
1. `vdg devices` / `vdg models` 列出已注册设备与负载。
2. `vdg simulate --device RTX5090 --model LTX_2_3 --scenario ltx_t2v_480p_81f` 跑单次仿真。
3. 输出 `latency / energy / quality / peak_memory / throughput / breakdown / pareto_tag`。
**预期结果（`results.json`）**：RTX 5090 / LTX-2.3 / 480p·81f·30 步 → 8.87s / 3949J / q84.0 / 5.24GB。
**价值**：在无硬件条件下预估部署可行性，替代"装环境试跑"的高成本试错。

### UC2 — 自动诊断 LTX-2.3 在 MPS 上的数值发散 + 运行时补丁落地

**触发**：LTX-2.3 在 Apple Silicon (MPS) 上出黑帧/NaN（用户实战 `MPS_BLACK_VIDEO_FIX.md` 场景）。
**主流程**：
1. `vdg probe --device M4_Max --precision bf16` 运行 `NumericalProbe`。
2. 探针在边界输入上测四个敏感 op（GELU / AdaLN / RMSNorm / Softmax）的 cpu-fp32 参考差。
3. 检测到发散 → `DiagnosticAgent` 推荐匹配修复技能（`adaln_fp32` 等）。
4. `RepairAgent` 产出 patched_config + 落地补丁指令（LTX 三处 cast fp32 模板）。
5. `vdg runtime --device M4_Max --model LTX_2_3 --scenario ltx_t2v_480p --runtime torch`
   发射**可执行补丁脚本**：`TorchRuntime.apply_all()` 在任意 nn.Module 上就地打 gelu_fp32/adaln_fp32
   （按名称/类名启发式定位 `transformer_blocks.*` 的 GELU/AdaLN 站点），`unpatch()` 回滚——
   修复从"grep 模板"升级为"可运行的进程内补丁"，无需手工改源码。
**预期结果（`results.json` probe_table，本机 M4 Max 真实 torch/MPS 实测）**：
M4_Max @ bf16 → `adln_modulate=divergence`，建议 `adaln_fp32`；@ fp16 → `rmsnorm=divergence`，建议 `rmsnorm_fp32`。
**价值**：把"黑屏根因定位"从人工 grep 模型代码自动化为探针+规则；修复指令直接给出
`grep -n 'scale_msa\|gate_msa\|gelu' comfy/ldm/lightricks/model.py` 的定位点与三处 cast 模板，
并可用 `vdg runtime --runtime torch` 直接发射可执行的 `TorchRuntime` 补丁脚本（含 `unpatch()` 回滚）。

### UC3 — 能耗预算下自动选择加速组合

**触发**：给定能耗预算，要求平台自动挑出"既满足 SLO 又不超能耗"的技能组合。
**主流程**：
1. `vdg govern --device RTX5090 --model LTX_2_3 --scenario ltx_t2v_480p_81f --energy-budget 4000`。
2. `RuleEngine` R2 检测基线能耗超预算 → `preferred_skills` 加入 `step_distill`（最大能耗杠杆）。
3. `AccelSelectorAgent` 枚举单技能变体（含可配置技能的全部变体）+ 显式对/三元组 + 5 个命名配方（RTX 5090 上 46 个组合，M4 Max 因 R1 禁用 CUDA 注意力技能降至 24 个），逐一仿真，按
   (latency, energy, quality) 三目标 Pareto 排序，policy 验收过滤。
4. 输出 top 组合 + 排名备选 + 违规列表。
**预期结果（`results.json` governance_runs）**：RTX5090 / 480p·81f → top=`teacache+sage_attention`，
final 1.95s / 828J / q86.1，feasible=True，46 个备选；技能单扫中 `step_distill:4step` 1.60s/712J（5.54×，最大能耗杠杆）。
**价值**：把"该用哪组加速"从经验试错变成 Pareto 全量枚举+约束验收的结构化决策。

### UC4 — 长视频内存规划

**触发**：生成 1025 帧（~43s）长视频，单次去噪+VAE 解码显存爆掉。
**主流程**：
1. `vdg simulate --device RTX5090 --model LTX_2_3 --scenario long_video_1025f --skills vae_tiling,offload`。
2. 仿真器报 OOM 风险（peak_memory 21.04GB < 32GB 但 VAE 段激活可能爆）。
3. `vae_tiling`（memory_ratio 0.25，HunyuanVideo 32GB→8GB）+ `offload`（block-swap 0.5）把峰值压到设备上限内。
4. 治理管道选 `sage_attention:v3` 组合（`results.json`：long_video top 组合，148.9s→31.7s / 66257J→10462J）。
**价值**：用上下文窗口（Kijai 81+16 重叠）+ VAE 分块+offload 把"不可运行"变"可运行"，无需升级硬件。

### UC5 — 训推异构导出分档

**触发**：训练侧产出一个 checkpoint，需按端侧设备能力分档导出（数据中心 fp8 / 消费 NV GGUF / 边缘 NPU int8）。
**主流程**：
1. 对同一负载（如 LTX-2.3）在多设备上跑 `vdg govern`，对比 Pareto 前沿与 policy 验收。
2. `RuleEngine` R3：质量目标 >85 → 限制量化只能 `gguf_q4`（禁 nvfp4/int8）；R4：int8-only 设备
   （Ampere/Jetson）部署 fp8 训练权重 → 加 boundary-block bf16 保护（首 2 + 末 3 块）。
3. 按设备类别输出分档导出建议：DC→fp8 原权；消费 NV→GGUF Q4 或 NVFP4（5090）；NPU→int8 + boundary 保护。
**价值**：把"导出分档"与"鲁棒性修复"耦合——同一量化在不同设备上回报相反（消费 NV NVFP4 既快又省，
NPU int8 需 boundary 保护），平台据此给出设备特定的导出+修复方案，而非一刀切量化。

### UC6 — 一键生成 ComfyUI 可执行工作流

**触发**：治理管道已选定加速组合，想把决策直接变成 ComfyUI 里能跑的图，而不是手工对照节点搭。
**主流程**：
1. `vdg govern --device RTX5090 --model LTX_2_3 --scenario ltx_t2v_480p_81f` 跑治理闭环，选出 top 组合。
2. `vdg runtime --device RTX5090 --model LTX_2_3 --scenario ltx_t2v_480p_81f --runtime comfyui --out wf.md`
   发射产物：`/prompt` API 格式工作流 JSON + 节点清单 + 依赖说明（ComfyUI-TeaCache / VideoHelperSuite /
   ComfyUI-GGUF pack）。
3. `--comfy-checkpoint/unet/vae/clip` 指定模型路径；`--prompt-positive/--prompt-negative` 覆盖提示词。
4. POST 到 `http://127.0.0.1:8188/prompt`（body `{"prompt": <json>}`）或 Load (API Format) 直接执行。
**预期结果**：RTX 5090 / LTX-2.3 输出 8 节点工作流 `CheckpointLoaderSimple → CLIPTextEncode×2 /
EmptyLTXVLatentVideo → TeaCache(rel_l1_thresh=0.1) → KSampler(steps=30) → VAEDecode → VHS_VideoCombine`；
修复技能自动转为 `render_patch_script()` 的可执行 torch 补丁段而非节点（修复是进程内 patch）。
**价值**：治理闭环的最后一公里——从"决策列表"到"ComfyUI 可执行产物"一键直达；配置缺失（如
comfyui+teacache 缺 `rel_l1_thresh`）在 `RuntimeEnvelope.validate()` 处 fail-fast，不产生坏工作流。

---

## 4. 用例–设备–能耗映射矩阵

| 用例 | 消费 NV | Apple Silicon | 边缘 NPU | 数据中心 |
|---|---|---|---|---|
| UC1 profile+simulate | ✅ 主 | ✅ | ✅ | ✅ |
| UC2 数值诊断 | (simulated) | ✅ 真实 MPS 实测 | (simulated) | (simulated) |
| UC3 能耗预算选加速 | ✅ 主 | ✅（带宽受限，skill 受限） | ✅ | ✅ |
| UC4 长视频内存规划 | ✅ 主 | ✅（统一内存优势） | ✅（显存紧） | ✅ |
| UC5 训推导出分档 | ✅ NVFP4/GGUF | ✅ GGUF | ✅ int8+boundary | ✅ fp8 原权 |
| UC6 一键生成 ComfyUI 工作流 | ✅ 主 | ✅ | ✅（Jetson ComfyUI） | ✅ |

> 表中 (simulated) 指 `NumericalProbe` 在无该真机时回退到 numpy 仿真报告（仍按报告阈值计算，非硬编码）。

---

## 5. 能耗约束条件

每类设备的能耗约束由 `Scenario.energy_budget_j`（场景 SLO）+ `Policy.energy_budget_j`（CLI `--energy-budget` 覆盖）
共同定义，`Policy.enforce` 在每个候选仿真结果上检查 `energy_j > energy_budget_j` 即记违规。

**按设备类别的能耗约束特征（LTX-2.3 / 480p·81f，`results.json`）**：

| 设备类别 | 代表 | 30 步基线能耗 | 4 步蒸馏能耗 | 能耗约束特征 |
|---|---|---|---|---|
| 消费 NV | RTX 5090 / 4090 | 3949J / 3930J | 3501J / 3485J | 高功耗(450–575W)×短延迟；蒸馏能耗杠杆 ~5.5×；典型 budget 5–50kJ |
| Apple Silicon | M4 Max | 24959J | 22131J | 系统功耗高(≤480W)×长延迟(带宽受限)；能耗最重；需蒸馏+缓存大幅压缩 |
| 边缘 NPU | Jetson Thor T5000 | 1440J | 1277J | 低功耗(40–130W)×中等延迟；单 clip 能耗最低；典型 budget 20kJ |
| 数据中心 | H100/B200 | (训练侧) | — | 训练能耗是单 clip 推理的 ~10^5–10^6 倍；约束在 TCO/天级别 |

**关键结论**：
- **能耗 = 功耗 × 延迟**。M4 Max 虽然系统功耗与 5090 相当，但带宽瓶颈把延迟拉到 68.85s（5090 的 7.8×），
  导致单 clip 能耗是 5090 的 6.3×。这解释了为何"带宽是 Apple Silicon 的真正能耗杠杆"——提速即省电。
- **步数蒸馏是跨设备通用能耗杠杆**：5090 3949J→712J（`step_distill` 单技能，5.54×），
  因延迟随步数近线性下降而功耗不变。这是"训练侧一次性投入，推理侧持续省电"的范式。
- **NPU 的能耗优势来自低功耗而非高速度**：Thor 1440J < 5090 的 3949J，但延迟 14.4s > 5090 的 8.87s——
  能耗约束在边缘场景天然宽松，约束点在延迟 SLO 与显存。
