# VDG 系统设计整体思路与建模方法可演进性

> 交付件对应：AI 负载建模子方向「系统设计整体思路（一页）+ 建模方法的可演进性（一页）」。
> 主模型：LTX-2.3（Lightricks 视频 DiT，2B 参数，高压缩 3D VAE 8×32×32）。

本文分两部分：**Part 1** 描述系统设计目标与思路；**Part 2** 描述建模方法的可演进性。
所有量化数字来自 `results.json`（由 `scripts/gen_results.py` 调用真实仿真器产出，可复现）。

---

## Part 1 — 系统设计整体思路

### 1.1 目标

VDG（Video Diffusion Governance，视频扩散治理）的目标是：**在训练侧与推理侧设备高度异构的条件下，
为"视频 DiT 部署到端侧"这一决策提供可运行、可复现、数据落地的性能–能耗建模与 AI 治理平台**，
直接回答"某模型在该设备上能否跑、跑多快、耗多少电、画质掉多少、该用哪组加速技能"。

平台不是论文级的 bit-exact 仿真器，而是**工程级近似仿真器**：所有常量来自一手调研报告且在代码内标注来源，
误差区间与已知数据缺口显式标记，从而在工程可接受精度内给出可指导部署的结论。

### 1.2 核心问题：训推异构的"精度悬崖"

当前开源视频生成已收敛到 **DiT + Flow Matching + 3D 时空 VAE** 范式。训练在 NVIDIA 数据中心以
bf16 为主、fp8/NVFP4 为新趋势完成；产出的 checkpoint（safetensors）需部署到算力/精度/架构差异极大的端侧：

| 维度 | 训练侧（NVIDIA DC） | 端侧（MPS / 消费 NV / NPU） | 鸿沟 |
|---|---|---|---|
| 算力精度 | bf16 为主；Hopper/Blackwell fp8/NVFP4 | MPS bf16/fp16；消费 NV fp8/fp4/int8；NPU int8/int4 | 训练从容、端侧逼近下限 |
| 注意力后端 | FA-3（H100 740 TFLOPS）/ fp8 attention | MPS 无融合内核（math 回退）；消费 NV SageAttention；NPU 需原生移植 | MPS 注意力无加速且数值风险高 |
| 显存/内存 | HBM3e 80–192GB / 3.35–8 TB/s | MPS 统一内存 64–512GB / 410–819 GB/s；消费 NV 24–32GB；NPU 64–128GB / 273 GB/s | 端侧容量够但带宽差 4–10× |

异构的本质矛盾不是"算力不够"，而是**精度悬崖**：同一模型被推向精度下限时，VAE 是画质硬下限
（fp8 VAE 即可见伪影），AdaLN 调制的 `(1+scale)` 灾难性抵消是最隐蔽的崩溃源——
用户的 MPS 黑屏修复（`MPS_BLACK_VIDEO_FIX.md`）正是这条悬崖上的典型实例，且 2026 年仍在复现
（ComfyUI #15315：M4 Max 上全黑+音频 NaN）。

### 1.3 VDG 方法：Roofline + 能耗 + AI 治理三层

VDG 把"端侧部署决策"建模为三层闭环：

1. **Roofline 性能层**（`core/roofline.py`）——把每个去噪步拆成 attention / FFN 两段，分别用
   `achievable = min(peak_flops, arithmetic_intensity × mem_bw)` 求可达算力，再除 FLOPs 得时间。
   Attention 段按注意力后端映射到对应精度峰值（sage3→FP4、sage2→FP8、sage1→INT8、math→0.5×峰值）；
   FFN 段用步数精度峰值。VAE 解码与文本编码各跑独立 roofline。这套抽象把"设备差异"压缩成
   `(peak_flops[precision], mem_bw)` 两元组，使同一模型在不同设备上的预测只换两个数。

2. **能耗层**（`core/energy_model.py`）——可插拔能耗模型把计算时间转焦耳：`TDPEnergyModel` 用
   `power = idle + (tdp - idle) × utilization` 线性外推；`MeasuredEnergyModel` 在有 pynvml 的 NVIDIA
   主机上读实时功率积分，否则透明回退到 TDP。加速技能的 `energy_ratio` 刻画"速度之外"的能效增益
   （如 FP4 tensor core 既快又省电，`energy_ratio=0.7`）。

3. **AI 治理层**（`agents/` + `governance/`）——四个治理代理串成闭环：
   `DiagnosticAgent`（数值探针+修复建议）→ `RuleEngine`（设备/质量/能耗四条硬规则）→
   `AccelSelectorAgent`（枚举技能组合、仿真、Pareto 排序）→ `RepairAgent`（落地修复补丁+指令）→
   `SimulatorAgent`（最终权威仿真）。治理代理用 `GovernanceDecision` 结构化输出"用哪个技能、
   什么配置、为什么"，而非黑盒推荐。

### 1.4 脚手架：与 diffusers / ComfyUI 生态对齐

VDG 不重新发明运行时，而是**复刻 ComfyUI/pytest 式的装饰器注册表**（`core/registry.py`）：
设备/负载/技能/能耗模型用 `@register_device` / `@register_load` / `@register_skill` /
`@register_energy_model` 自注册，导入 `vdg` 即自动发现。每个技能的 `apply()` 返回一个
`runtime_envelope`——目标运行时（ComfyUI / diffusers / LightX2V / MLX / TensorRT）消费的配置字典，
使 VDG 的治理结论能直接落到用户已有的 ComfyUI 工作流或 diffusers 管线。这与用户的真实工作流
（LTX-2.3 on MPS via ComfyUI）同构。

### 1.5 数据落地：以 LTX-2.3 为锚

平台以 **LTX-2.3 为首要负载（hero load）**，并内置 Wan 2.1/2.2、HunyuanVideo、CogVideoX、Open-Sora 2.0
作为跨模型可比性参照。LTX-2.3 的全部架构数字直接取自 HuggingFace `config.json` / safetensors 元数据：
28 层、hidden_dim=2048、32 头、FFN 4×、VAE 压缩 (8,32,32)、1.923B 参数、419.2M VAE 参数、T5-XXL 4.76B 文本编码器。
设备规格取自一手报告与 NVIDIA 公开 dense-tensor 规格（如 RTX 5090 FP4-sparse 3352 AI TOPS → dense 1676 TFLOPS）。

**仿真锚点（`results.json`，LTX-2.3 / 480p·81f）**：

| 设备 | 30 步基线 | 4 步蒸馏 | 带宽 |
|---|---|---|---|
| RTX 5090 | 8.87s / 3949J | 7.87s / 3501J | 1.79 TB/s |
| RTX 4090 | 11.27s / 3930J | 9.99s / 3485J | 1.01 TB/s |
| Jetson Thor T5000 | 14.40s / 1440J | 12.77s / 1277J | 273 GB/s |
| M4 Max | 68.85s / 24959J | 61.05s / 22131J | 546 GB/s |

数字印证了报告的核心洞察：**MPS 不是算力瓶颈而是带宽瓶颈**（68.85s vs 5090 的 8.87s，差距远超算力比）；
**Jetson Thor 算力充足但 273 GB/s 带宽拖累**（14.4s）；**能耗与功耗强相关**（M4 Max 480W 系统 → 24959J，
Thor 130W → 1440J）。

---

## Part 2 — 建模方法的可演进性

### 2.1 四个可插拔维度

VDG 的抽象把"视频 DiT 部署"拆成四个正交可插拔维度，每个维度都是一个 `Registrable` 子类 + 装饰器：

| 维度 | 基类 | 装饰器 | 实例数（已内置） | 扩展动作 |
|---|---|---|---|---|
| 设备 Device | `DeviceProfile` | `@register_device` | 13（Apple/NV DC/消费 NV/Jetson/NPU） | 新增一个 `spec()` 返回 `DeviceSpec` 的子类 |
| 负载 Load | `LoadModel` | `@register_load` | 9（LTX-2.3 + Wan/Hunyuan/Cog/OpenSora） | 新增一个 `characteristics()` 返回 `VideoDiTLoad` 的子类 |
| 技能 Skill | `Skill` | `@register_skill` | 12（7 accel + 5 repair） | 新增 `predict()` 返回 `SkillImpact` + `apply()` 返回 runtime envelope |
| 能耗模型 | `EnergyModel` | `@register_energy_model` | 2（TDP / Measured） | 新增 `energy()` 实现 |

**关键设计**：基类已用 roofline 模型实现 `tokens_for` / `per_step_flops` / `memory_footprint`，
子类**只填数据**（`characteristics()` 返回 dataclass），不重写算法。这让"加一个新模型"退化成
"填一张架构参数表"，是平台可演进性的根基。

### 2.2 如何新增一个设备 / 负载 / 技能

以新增设备为例（完整范式见 `src/vdg/devices/`）：

```python
from ..core.contracts import DeviceCategory, DeviceProfile, DeviceSpec
from ..core.registry import register_device

@register_device
class MyNPU(DeviceProfile):
    def spec(self) -> DeviceSpec:
        return DeviceSpec(
            name="MyNPU", category=DeviceCategory.EDGE_NPU,
            memory_gb=32.0, memory_bandwidth_gbps=200.0,
            compute_tflops={"fp16": 100.0, "int8": 200.0},
            tdp_w=60.0, idle_power_w=6.0,
            supported_precisions=["fp16", "int8"],
            attention_backends=["vendor_attn", "math"],
            unified_memory=True,
        )
    def is_available(self) -> bool: ...   # 探测真机，绝不抛异常
    def measure_power(self) -> float | None: ...
```

把文件放进 `vdg/devices/`，`import vdg` 即自动注册，仿真器与治理代理立即可用——**无需改动任何中央代码**。
新增负载/技能同理（分别见 `loads/video_dit.py`、`skills/accel/*.py`）。技能额外需声明
`applies_to`（`DeviceCategory` 列表）与 `kind`（`"accel"` / `"repair"`），治理代理据此自动筛选适用技能。

### 2.3 注册表的可扩展性机制

`Registry` 是 `(kind, name) -> class` 的扁平存储，`discover()` 顺序导入 `loads → skills → devices`
子包使装饰器执行自注册。两个稳健性设计保证"半装树不崩"：

- **导入失败吞没**：`_import_subpackages` 与 `discover` 都 `try/except`，缺可选插件模块时静默跳过，
  绝不让一个坏插件毒化整个注册表。
- **裸/命名双用例**：装饰器同时支持 `@register_device`（名字取类名）与 `@register_device("rtx_5090")`
  （显式命名），降低使用门槛。

可演进性还体现在**技能组合的可组合性**：`SkillImpact` 四元组（speedup / memory_ratio / quality_delta /
energy_ratio）以次乘性 `product(speedup)^0.85` 组合，指数 <1 建模瓶颈转移与作用域重叠的收益递减
（如蒸馏降步后 attention 占比下降，SageAttention 的相对增益随之缩小）。新增技能只需给出自己的
`SkillImpact`，组合数学自动生效，无需改动仿真器。

### 2.4 与 DSX / OpenUSD 数字孪生的映射

VDG 的设备–负载–能耗抽象与 NVIDIA Omniverse **DSX Blueprint**（AI 工厂数字孪生）的
OpenUSD 场景图同构，可直接作为 DSX 的"算力负载层"输入：

| VDG 概念 | DSX / OpenUSD 对应 | 映射含义 |
|---|---|---|
| `DeviceSpec`（memory_gb / mem_bw / tdp_w / compute_tflops） | USD `Asset` / 设备 Prim 属性 | 单点设备规格 → 数字孪生资产属性 |
| `SimulationResult.energy_j` / `latency_s` | DSX 电气负载 / CFD 热仿真输入 | 单次推理的焦耳与秒 → 数据中心能耗–热仿真的事件源 |
| `Policy`（energy_budget / latency_slo / max_memory） | DSX SLO / 容量规划约束 | 治理验收包络 → 工厂级 SLA 约束 |
| `Scenario`（分辨率/帧数/步数） | USD `Stage` 中的负载事件 | 生成任务 → 时序负载事件 |
| 设备类别（4 类） | DSX 机架/节点分组 | 设备族群 → 数据中心拓扑分组 |

映射价值：VDG 在单设备粒度给出"焦耳/clip"，DSX 在工厂粒度做"MW·h/天 + 散热 PUE"。
两者通过 `SimulationResult.energy_j` 这个公共量纲衔接——VDG 是 DSX 数字孪生的**负载建模前端**，
把"某个视频 DiT 在某设备上的能耗"变成 DSX CFD/电气仿真的可消费输入。这使得 VDG 的建模结论
可从"单卡部署决策"平滑升级到"数据中心级能耗–热仿真"。

### 2.5 已知建模缺口与演进路径

平台显式标记的数据缺口（诚实标注，便于演进校准）：

- Apple Silicon 无公开 TFLOPS（`compute_tflops` 为保守估计，带宽才是真瓶颈）；
- VAE 解码 FLOPs 用几何均值金字塔近似（精确 per-video VAE FLOPs 报告标记为数据缺口）；
- 技能组合倍率为有据保守估计（`COMBINATION_EXPONENT=0.85`，可随实测校准）；
- Jetson Thor 视频 DiT 延迟无公开基准（scenario SLO 为规划目标，显式标注）。

演进路径：接入 `MeasuredEnergyModel` 的实测功率积分可逐步替代 TDP 外推；技能 `apply()` 的真实内核补丁
（如 LTX 三处 cast fp32 已在 `skills/repair/adaln_fp32.py` 落地）可把仿真预测校准到实测；
新增 `EnergyModel` / `DeviceProfile` 子类即可对接 DSX OpenUSD 场景图，无需改动核心。
