# MPS 黑图/NaN 排查与修复工作流

> 适用于 Apple Silicon (MPS) 上 ComfyUI 生成全黑视频 / NaN 的问题。
> 基于 LTX-2.3 实战修复经验，可复用于 MiniMax H3、Wan 等其他 DiT 模型。

---

## 1. 症状识别

出现以下任一症状即可能是 MPS 数值精度问题：

| 症状 | 表现 |
|------|------|
| **全黑视频** | 输出视频每一帧纯黑，VAE 解码结果全 0 |
| **NaN 传播** | 日志出现 `invalid value encountered in cast` 或 `RuntimeError: NaN` |
| **特定精度才炸** | `--force-fp32` 正常，`--fp16-unet` 或 `--bf16-unet` 黑屏 |
| **特定平台才炸** | 同一模型同一工作流在 NVIDIA 正常，MPS 黑屏 |

---

## 2. 排查决策树

```
黑图/NaN
  │
  ├─ Step 1: 模型完整性
  │   └─ TE 是否包含 vision_model 权重？
  │      （GGUF TE 常缺失 vision_model → I2V 黑屏）
  │      验证: python -c "from safetensors import safe_open; ..."
  │
  ├─ Step 2: 启动参数
  │   └─ 用 --force-fp32 测试是否正常？
  │      ✅ 正常 → 确认是精度问题，进入 Step 3
  │      ❌ 仍然黑 → 模型/工作流问题，不是精度问题
  │
  ├─ Step 3: 隔离问题精度
  │   └─ 分别测试:
  │      --fp16-unet  → 黑？
  │      --bf16-unet  → 黑？
  │      --force-fp32 → 正常？
  │      （确定是 fp16 炸、bf16 炸、还是都炸）
  │
  ├─ Step 4: 排除第三方插件
  │   └─ 禁用 ComfyUI-AppleSilicon-FP8 / Spectrum 等插件
  │      仍然黑 → 确认是 ComfyUI/PyTorch MPS 本身的问题
  │
  └─ Step 5: 定位 NaN 来源（见第 3 节）
      └─ GELU? AdaLN? attention? 逐个排查
```

---

## 3. 根因定位（实测方法）

### 3.1 对比 CPU vs MPS

核心思路：**同一个 op，CPU 结果正确但 MPS 结果错误 = PyTorch MPS 内核 bug**。

```python
import torch, warnings
warnings.filterwarnings("ignore")

x_vals = [5, 10, 15, 20, 30, 40]

for dt in [torch.float16, torch.bfloat16]:
    print(f"\n=== {dt} ===")
    for val in x_vals:
        x_cpu = torch.tensor([val], dtype=dt)
        x_mps = torch.tensor([val], dtype=dt, device="mps")

        gelu_cpu = torch.nn.functional.gelu(x_cpu, approximate="tanh")
        gelu_mps = torch.nn.functional.gelu(x_mps, approximate="tanh")

        status = "❌ NaN" if torch.isnan(gelu_mps).item() else "✅"
        print(f"  GELU({val:>3})  CPU={gelu_cpu.item():.2f}  MPS={gelu_mps.item():.2f}  {status}")
```

### 3.2 逐项拆解公式

如果融合 op 产生 NaN，手动逐项计算（走独立 op）来定位：

```python
# GELU tanh 近似: 0.5*x*(1+tanh(sqrt(2/pi)*(x + 0.044715*x³)))
x = torch.tensor([15.0], device="mps", dtype=torch.bfloat16)

step1 = x * x * x               # x³
step2 = 0.044715 * step1        # 0.044715 * x³
step3 = x + step2               # x + ...
step4 = (2/torch.pi)**0.5 * step3
step5 = torch.tanh(step4)
step6 = 1 + step5
step7 = 0.5 * x * step6         # 最终结果

print(f"手动逐项: {step7.item()}")      # 正确 → 融合内核有 bug
print(f"F.gelu 融合: {torch.nn.functional.gelu(x, approximate='tanh').item()}")  # NaN
```

**判断规则**：
- 手动逐项正确 + 融合 op NaN → **融合 Metal 内核 bug**
- 手动逐项也 NaN → **数值范围溢出**（如 fp16 的 x³ 在 |x|>40 溢出 65504）

---

## 4. 两个经典 NaN 陷阱

### 陷阱 1：GELU 融合内核 bug（bf16）

| 精度 | 机制 | 阈值 |
|------|------|------|
| fp16 | x³ 超过 65504 → Inf → NaN | \|x\| > 40 |
| bf16 | MPS 融合 GELU Metal 内核缺陷（非溢出） | \|x\| ≥ 15 |

> bf16 的指数范围和 fp32 相同（8 位），x³ 不会溢出。
> 但 MPS 的融合 GELU 内核在 bf16 下有实现缺陷，x≥15 即 NaN。
> CPU 的 bf16 GELU 正常，证明是 MPS 特有 bug。

### 陷阱 2：AdaLN 灾难性抵消

```python
# AdaLN: rms_norm(x) * (1 + scale) + shift
# 当 scale ≈ -1 时，bf16 的 7 位尾数让 (1+scale) 丢失精度

scale = -0.9999
# fp32: 1 + (-0.9999) = 0.0001   ← 有精度
# bf16: 1 + (-0.9999) = 0.0      ← 灾难性抵消，丢全部有效数字
```

---

## 5. 修复模式

### 5.1 标准修复：MPS 上 cast 到 fp32

参考 Hunyuan3D / HiDream 已有的 MPS workaround：

```python
# Hunyuan3D 的做法 (comfy/ldm/hunyuan3dv2_1/hunyuandit.py:16-17)
if gate.device.type == "mps":
    return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)
```

### 5.2 三处必改位置

在 DiT 模型的 `model.py` 里，以下三处在 MPS 上必须 cast fp32：

#### ① GELU

```python
# 修改前
x = F.gelu(x, approximate="tanh")

# 修改后
if x.device.type == "mps":
    x = F.gelu(x.float(), approximate="tanh").to(dtype=x.dtype)
else:
    x = F.gelu(x, approximate="tanh")
```

#### ② AdaLN 自注意力调制

```python
# 修改前
attn_out = self.attn1(rms_norm(x) * (1 + scale_msa) + shift_msa, ...) * gate_msa
x += attn_out

# 修改后
if x.device.type == "mps":
    scale_msa, shift_msa, gate_msa = [t.float() for t in (scale_msa, shift_msa, gate_msa)]
    x_fp32 = x.float()
    attn_out = self.attn1(rms_norm(x_fp32) * (1 + scale_msa) + shift_msa, ...) * gate_msa
    x = (x_fp32 + attn_out).to(dtype=x.dtype)
else:
    attn_out = self.attn1(rms_norm(x) * (1 + scale_msa) + shift_msa, ...) * gate_msa
    x += attn_out
```

#### ③ AdaLN MLP 调制

```python
# 修改前
y = rms_norm(x)
y = y * (1 + scale_mlp) + shift_mlp   # 或 addcmul
x = x + self.ff(y) * gate_mlp

# 修改后
if x.device.type == "mps":
    scale_mlp_f, shift_mlp_f, gate_mlp_f = [t.float() for t in (scale_mlp, shift_mlp, gate_mlp)]
    y_f32 = rms_norm(x.float())
    y_f32 = y_f32 * (1 + scale_mlp_f) + shift_mlp_f
    x = x + self.ff(y_f32.to(dtype=x.dtype)) * gate_mlp_f.to(dtype=x.dtype)
else:
    y = rms_norm(x)
    y = y * (1 + scale_mlp) + shift_mlp
    x = x + self.ff(y) * gate_mlp
```

### 5.3 查找待修位置

```bash
# 找模型文件
find comfy/ldm -name "model.py" -path "*/<model_name>/*"

# 找 GELU 调用
grep -n "gelu\|approximate" comfy/ldm/<model_name>/model.py

# 找 AdaLN 调制（scale/shift/gate 模式）
grep -n "scale_msa\|shift_msa\|gate_msa\|scale_mlp\|shift_mlp\|gate_mlp" comfy/ldm/<model_name>/model.py

# 参考其他模型已有的 MPS workaround
grep -rn "device.type.*mps\|\.float()" comfy/ldm/hunyuan3dv2_1/ comfy/ldm/hidream/
```

---

## 6. 验证清单

- [ ] `--force-fp32` 能正常出图（确认模型本身没问题）
- [ ] `--fp16-unet` 修复前黑屏（确认问题存在）
- [ ] 打补丁后 `--fp16-unet` 正常出图（确认修复有效）
- [ ] 日志无 `invalid value encountered in cast`
- [ ] 生成的视频非全黑（肉眼检查 + 像素值检查）
- [ ] 对比修复前后速度（fp16 应比 fp32 快约 2×）

```python
# 快速验证输出非全黑
import torch
latent = torch.load("output_dir/xxx.latent")  # 或从 VAE 输出取
print(f"全零? {latent.abs().sum().item() == 0}")
print(f"NaN?  {torch.isnan(latent).any().item()}")
print(f"范围: [{latent.min().item():.4f}, {latent.max().item():.4f}]")
```

---

## 7. 已知模型状态

| 模型 | MPS workaround | 状态 |
|------|---------------|------|
| Hunyuan3D | ✅ 已有（`hunyuandit.py:16-17`） | 官方已修 |
| HiDream | ✅ 已有（时间步用 fp32） | 官方已修 |
| LTX-2.3 | ✅ 已修（本 fork `mps-fp16-fix` 分支） | 本地修复，可 PR |
| MiniMax H3 | ❌ 待修（需先升级 ComfyUI 到 0.30.0+） | 下一步 |

---

## 8. 快速复用模板

遇到新模型黑屏时，按此顺序执行：

```bash
# 1. 确认是精度问题
python main.py --force-fp32 ...        # 正常？
python main.py --fp16-unet ...         # 黑屏？

# 2. 运行根因定位脚本（见第 3 节）

# 3. 找到模型文件里的 GELU 和 AdaLN
grep -n "gelu\|scale_msa\|gate_msa" comfy/ldm/<model>/model.py

# 4. 按 5.2 模式打补丁（if mps → fp32）

# 5. 验证
python main.py --fp16-unet ...         # 正常了？
```
