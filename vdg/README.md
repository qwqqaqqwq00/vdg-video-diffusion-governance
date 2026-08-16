# VDG — Video Diffusion Governance

视频扩散治理平台：为「视频 DiT 训练在 NVIDIA 数据中心、推理部署到 Apple Silicon / 消费级 NVIDIA / 工业边缘 NPU」
这一训推异构场景，提供**可运行、可复现、数据落地**的性能–能耗建模与 AI 治理。

主模型 **LTX-2.3**（Lightricks 视频 DiT，2B 参数，高压缩 3D VAE 8×32×32）——即用户已备好并在 Apple Silicon (MPS)
上实战修复过的模型。内置 Wan 2.1/2.2、HunyuanVideo、CogVideoX、Open-Sora 2.0 作为跨模型可比性参照。

- 仿真核心：roofline 性能模型 + 可插拔能耗模型 + 次乘性技能组合数学（`numpy` + 标准库，Python 3.10+）；`calibration.py` 用真实测量锚点标定仿真（`vdg calibrate`）。
- 治理闭环：诊断 → 规则 → 选择 → 修复 → 仿真，输出结构化决策、Pareto 备选与落地补丁指令。
- 生态对齐：ComfyUI/pytest 式装饰器注册表（17 设备 / 9 负载 / 19 技能）；`runtime/` 包把治理决策发射为**可执行运行时产物**（ComfyUI 工作流 JSON / diffusers 脚本 / torch 补丁 / LightX2V / MLX 命令）。

## 安装

```bash
cd /Users/wrd/Documents/UE/Agents/vdg
pip install -e .
```

依赖：Python 3.10+、`numpy`。`torch`/`pynvml` 为可选：有 torch+MPS/CUDA 时 `NumericalProbe` 跑真实测量，
否则透明回退到 numpy 仿真（平台在任何机器上可运行）。

## 快速上手（3 个 CLI 示例，均以 LTX_2_3 为主模型）

### 1) 单次性能–能耗仿真

```bash
python -m vdg simulate \
  --device RTX5090 --model LTX_2_3 --scenario ltx_t2v_480p_81f \
  --precision bf16 --attention flash
```

输出 `latency / energy / quality / peak_memory / throughput / breakdown / pareto_tag`。
基准预测（`results.json`）：RTX 5090 / LTX-2.3 / 480p·81f·30 步 → **8.87s / 3949J / q84.0 / 5.24GB**。
加技能示例：追加 `--skills step_distill,teacache`（蒸馏+缓存组合）。

### 2) 全治理管道（能耗预算下自动选加速组合）

```bash
python -m vdg govern \
  --device RTX5090 --model LTX_2_3 --scenario ltx_t2v_480p_81f \
  --energy-budget 4000
```

枚举约 46 个技能组合（单技能变体 + 对/三元组 + 5 命名配方），逐一仿真后按 (latency, energy, quality)
Pareto 排序，policy 验收，输出 top 组合、排名备选、修复建议与落地补丁指令。
治理后预测：top=`teacache+sage_attention`，**1.95s / 828J / q86.1，feasible（4.5x）**。

### 3) 跨设备数值鲁棒性探针（LTX-2.3 在 MPS 黑屏根因定位）

```bash
python -m vdg probe --device M4_Max --precision bf16
```

在 GELU / AdaLN / RMSNorm / Softmax 四个敏感 op 的边界输入上对比 cpu-fp32 参考。
本机 M4 Max 实测结果（`results.json` probe_table）：@bf16 → `adln_modulate=divergence`（建议 `adaln_fp32`）；
@fp16 → `rmsnorm=divergence`（建议 `rmsnorm_fp32`）。这正是用户 `MPS_BLACK_VIDEO_FIX.md` 三处 cast fp32 修复的根因。

> 探针双路径：默认走**实测**（有 torch+MPS/CUDA 时真机测量），`--sim-probe` 强制走**仿真**
> （编码调研报告第 8 节的文档化阈值，主机无关、可复现）。本机 torch 2.12 已修复 MPS bf16 GELU
> 融合内核缺陷（x≥15 不再 NaN），故实测只报 AdaLN；`--sim-probe` 则同时报 GELU+AdaLN，
> 复现用户原始修复场景。`vdg govern` 默认用仿真探针做可复现规划，`--real-probe` 切到实测。

> 其它命令：`vdg devices`、`vdg models`、`vdg report`（插件清单+场景+配方总览）、`vdg calibrate`（锚点标定）、`vdg runtime`（发射运行时产物）、`vdg -h`。

## 杀手级演示：M4 Max + LTX-2.3 端到端治理

一条命令完整跑通「训推异构」治理闭环——这正是用户备好并在 MPS 上实战修复过的模型：

```bash
python -m vdg govern --device M4_Max --model LTX_2_3 --scenario ltx_t2v_480p
```

该命令依次完成四件事，全部数据落地、可复现：

| 步骤 | 结果 |
|------|------|
| **(a) 检测 MPS 低精度** | NumericalProbe（仿真阈值路径）报 2/4 敏感 op 发散：`gelu_tanh=nan`（MPS bf16 融合内核缺陷 \|x\|≥15）、`adln_modulate=divergence`（(1+scale) 灾难性抵消，-0.999→-1.0） |
| **(b) 推荐修复技能** | `gelu_fp32` + `adaln_fp32`——即 `MPS_BLACK_VIDEO_FIX.md` 三处 cast fp32 修复（GELU / AdaLN-attn / AdaLN-MLP），并输出可落地的 grep 定位 + 验证补丁指令 |
| **(c) 选择加速** | R1 规则禁用 `sage_attention`/`sliding_tile_attention`（CUDA+Triton only），并**偏好 `teacache`**（MPS 上唯一设备无关的注意力侧加速）；Pareto 选出 top=`teacache:thr0.2`（蒸馏在备选但质量不达标被排除） |
| **(d) 输出 SimulationResult** | baseline 68.85s/24959J → **final 28.57s/11525J（2.4x）/ q87.73 / feasible**，含 policy 验收、24 个排名备选、2 段补丁指令 |

> 加 `--real-probe` 切到真机实测：本机 torch 2.12 已修复 GELU 内核缺陷，实测只报 AdaLN（仍选 teacache、仍 feasible）——
> 平台如实区分「真机当前状态」与「文档化风险阈值」，这是工程级治理而非玩具脚本。

## 真实运行时对接

`vdg runtime` 把治理决策翻译成**可直接执行的运行时产物**（`src/vdg/runtime/`，全部 torch/diffusers 惰性导入，纯仿真环境不受影响），闭环不再止于仿真：

| 运行时 | 产物 | 消费方式 |
|---|---|---|
| `comfyui` | 完整 `/prompt` API 工作流 JSON + 节点说明 + 可执行 torch 补丁脚本 | POST 到 `http://127.0.0.1:8188/prompt` 或 Load (API Format) |
| `torch` | 进程内修复补丁脚本（`TorchRuntime`） | 对任意 nn.Module 就地打 gelu/adaln/rmsnorm/softmax/vae fp32 保护，`unpatch()` 回滚 |
| `diffusers` | LTX-Video 管道脚本 | 加载 HuggingFace 模型 → 就地应用修复 + 加速 API 映射 |
| `lightx2v` | `python -m lightx2v.infer` 启动命令 | 4 步蒸馏栈（NVFP4/SageAttention/TeaCache） |
| `mlx` | `mlx_video_generate` 启动命令 | Apple Silicon 统一内存推理 |

技能 `apply()` 产出的 envelope 升级为 `RuntimeEnvelope`（kind / target_runtime / 必需配置键校验）：
`validate()` 在运行时消费前 fail-fast 检查配置键（如 `comfyui+teacache` 必须带 `rel_l1_thresh`），配错的技能不会静默产生坏工作流。

示例 1 — RTX 5090 一键生成 ComfyUI 可执行工作流：

```bash
python -m vdg runtime --device RTX5090 --model LTX_2_3 \
  --scenario ltx_t2v_480p_81f --runtime comfyui --out comfyui_wf.md
```

产物为 8 节点 API 工作流：`CheckpointLoaderSimple → CLIPTextEncode×2 / EmptyLTXVLatentVideo → TeaCache → KSampler → VAEDecode → VHS_VideoCombine`（TeaCache 的 `rel_l1_thresh` 直接取自治理决策；`--comfy-checkpoint/unet/vae/clip` 可指定模型路径）。

**真实硬件冒烟测试**（`scripts/smoke_torch_runtime.py`）：在本机 torch + MPS/CUDA/CPU 上构造微型 toy DiT（nn.Linear + nn.GELU + LTX 风格 AdaLN 块 + RMSNorm + Softmax），跑 `TorchRuntime.apply_all()` 打 4 类修复补丁，断言 forward 仍可执行且有限、`_vdg_patched` 标记齐全、重放幂等、`unpatch()` 恢复全部原始 forward——证明 `TorchRuntime` 是真实进程内补丁而非发射器存根。真机 MPS + CPU 双跑通过，`tests/test_smoke_torch_runtime.py` 锁回归。

示例 2 — M4 Max 发射可执行修复补丁脚本：

```bash
python -m vdg runtime --device M4_Max --model LTX_2_3 \
  --scenario ltx_t2v_480p --runtime torch --out patch_script.py
```

产物为可直接运行的 `TorchRuntime.apply_all()` 脚本：加载你的 nn.Module，按名称/类名启发式定位
`transformer_blocks.*` 的 GELU/AdaLN 站点，就地打 `gelu_fp32`/`adaln_fp32`，并支持 `unpatch()` 回滚。

### 仿真校准（vdg calibrate）

`vdg calibrate --device H100 --model HunyuanVideo_13B --scenario ltx_t2v_720p_129f`：用真实测量锚点修正
roofline 预测，输出标定预测 vs 实测 + 相对误差表。实测效果：H100/HunyuanVideo-13B 基座低估 65.8% →
scale 2.92 修正；M4 Max/Wan2.1-1.3B 低估 92.3% → scale 12.92（Apple 带宽瓶颈的量化体现）。
`GovernancePipeline(simulator=CalibratedSimulator())` 注入即全链路标定。详见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §6。

## 架构（一段话）

VDG 分四层：**插件层**（`devices/loads/skills/` 用 `@register_*` 自注册，只填数据不写算法）→
**核心层**（`core/` frozen：`roofline` 算每步 attention/FFN 时间，`energy_model` 转焦耳，`simulator` 组合技能影响，
`calibration` 锚点标定，`registry` 做 ComfyUI 式 `(kind,name)->class` 发现）→ **治理层**（`agents/`+`governance/` 四代理闭环：
`DiagnosticAgent` 探针+修复建议 → `RuleEngine` 四条硬规则（R1 Apple 禁 CUDA 注意力 / R2 能耗超预算 prefer 蒸馏 /
R3 质量>85 限量化 / R4 int8-only 加 boundary 保护）→ `AccelSelectorAgent` 枚举+Pareto → `RepairAgent` 补丁指令 →
`SimulatorAgent` 最终权威仿真+policy 验收）→ **接入层**（`cli.py` + `runtime/` 把治理决策发射为可执行产物）。
核心不变量：蒸馏的步数下降只由技能 `speedup`（相对 `baseline_steps` 边际倍率）建模，绝不与 config 降步数叠加，避免双重计数。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) | 系统设计整体思路 + 建模方法可演进性（两页：目标/异构问题/三层方法/脚手架/数据落地 + 四可插拔维度/注册表扩展/DSX OpenUSD 数字孪生映射） |
| [docs/SYSTEM_USE_CASES.md](docs/SYSTEM_USE_CASES.md) | 设备上下文 + 6 个业务用例（UC1 profile+simulate / UC2 MPS 数值诊断+运行时补丁 / UC3 能耗预算选加速 / UC4 长视频内存规划 / UC5 训推导出分档 / UC6 一键生成 ComfyUI 可执行工作流）+ 组件图 + 能耗约束 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 架构图 + simulate→select→apply 闭环 + Pareto/5 命名配方 + 4 条业务结论 |
| [docs/TEST_REPORT.md](docs/TEST_REPORT.md) | 测试结构/方法论/鲁棒性分析/能耗量化对比（数据引用 `results.json`） |
| [results.json](results.json) | 本文档集的规范化数据源（`scripts/gen_results.py` 重生成）：场景矩阵/技能单扫/治理运行/探针表/能耗对比。注：仓库内 `test_results/results.json` 为并行测试流程的独立产物，与本文档集无关。 |

## 项目布局

```
vdg/
├── src/vdg/
│   ├── core/          # contracts / roofline / energy_model / scenario / simulator / calibration / registry  (frozen)
│   ├── devices/       # apple_silicon / nvidia_dc / consumer_nv / jetson / npu / detector       (17 devices)
│   ├── loads/         # video_dit.py  (LTX_2_3 + 8 reference loads)
│   ├── skills/
│   │   ├── accel/     # 13 技能: step_distill / teacache / sage_attention / quantization / offload / vae_tiling
│   │   │              #        / compile_graph / flash_attention / sliding_tile_attention / linear_attention
│   │   │              #        / mlx_sdpa / context_window / diffusion_forcing
│   │   └── repair/    # 6 技能: adaln_fp32 / gelu_fp32 / rmsnorm_fp32 / softmax_fp32 / vae_fp32 / boundary_block_bf16
│   │                  #        + numerical_probe
│   ├── agents/        # diagnostic / accel_selector / repair_agent / simulator_agent
│   ├── governance/    # pipeline / policy / rules
│   ├── runtime/       # envelope / torch_runtime / diffusers_runtime / comfyui_emitter / lightx2v_emitter / mlx_emitter
│   └── cli.py
├── tests/             # pytest 全绿（单元 + 治理仿真）
├── scripts/gen_results.py   # 重生成 results.json
├── docs/              # 4 份交付件文档
└── pyproject.toml
```

## 交付件映射（AI 负载建模子方向）

| 交付件要求 | VDG 对应 |
|---|---|
| 系统设计整体思路 + 建模方法可演进性（各一页） | `docs/SYSTEM_DESIGN.md` |
| 可运行测试代码（含性能-能耗仿真核心逻辑） | `src/vdg/core/` + `tests/`（pytest 全绿） + `scripts/gen_results.py` |
| 系统用例（设备上下文/关键用例/逻辑视图/业务用例/能耗约束） | `docs/SYSTEM_USE_CASES.md` |
| 架构设计（建模指导优化/性能-能耗权衡/业务结论） | `docs/ARCHITECTURE.md` |
| 测试报告（负载场景/仿真预测/鲁棒性/能耗量化对比） | `docs/TEST_REPORT.md` + `results.json` |

## 许可与数据来源

代码与文档为内部调研交付件。所有设备规格/模型架构/加速倍率/数值阈值均取自一手调研报告
（`端侧视频生成模型_训推异构方案调研报告.md` 及其四份子报告、`MPS_BLACK_VIDEO_FIX.md`），
并在源码 docstring 中标注来源 URL。仿真数字为平台建模预测（工程级近似，非 bit-exact），已知数据缺口在源码与文档中显式标记。
