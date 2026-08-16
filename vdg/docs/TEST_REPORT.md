# VDG Test Report -- Scenario Matrix

**Platform:** VDG (Video Diffusion Governance)
**Primary model:** LTX-2.3
**Scenarios:** 6
**Mode:** pure simulation (offline)

Every scenario runs the full GovernancePipeline (diagnose -> rules -> select accel -> repair -> simulate -> report). The governance layer auto-selects the best *policy-feasible* skill combo under each scenario's latency SLO, energy budget, and quality floor.

## Results

| # | Scenario | Device | Workload | Baseline lat (s) | Final lat (s) | Speedup | Energy (J) | Memory (GB) | Quality | Skills | Pareto | Feasible |
|--:|----------|--------|----------|-----------------:|--------------:|--------:|-----------:|------------:|--------:|--------|--------|:--------:|
| 1 | H100 + LTX-2.3 + 720p | H100 | [1280, 720] 129f | 14.56 | 4.74 | 3.076x | 2569 | 8.84 | 83.00 | step_distill:8step | efficient | yes |
| 2 | RTX 5090 + LTX-2.3 + NVFP4 | RTX5090 | [854, 480] 81f | 8.87 | 1.95 | 4.547x | 828 | 5.24 | 86.10 | teacache+sage_attention | repair:adaln_fp32 | efficient | yes |
| 3 | RTX 4090 + distill + LTX-2.3 + 480p | RTX4090 | [854, 480] 81f | 11.27 | 2.48 | 4.547x | 825 | 5.24 | 86.10 | teacache+sage_attention | repair:adaln_fp32 | efficient | yes |
| 4 | M4 Max + LTX-2.3 + 480p | M4_Max | [854, 480] 81f | 68.85 | 28.57 | 2.410x | 11525 | 5.24 | 87.73 | teacache:thr0.2 | repair:gelu_fp32+adaln_fp32 | efficient | yes |
| 5 | Jetson Thor + LTX-2.3 distill | Jetson_Thor_T5000 | [854, 480] 81f | 14.40 | 5.12 | 2.812x | 543 | 5.24 | 85.50 | step_distill:8step | repair:adaln_fp32 | efficient | yes |
| 6 | Ascend 910B + LTX-2.3 + INT8 | Ascend_910B | [854, 480] 81f | 11.62 | 4.32 | 2.692x | 1153 | 5.24 | 87.00 | step_distill:8step | repair:adaln_fp32+boundary_block_bf16 | efficient | yes |

## Energy Comparison

Baseline (unskilled) vs governance-final energy per scenario, with the percent energy saved. Negative values mean the final run used *more* energy (e.g. Apple Silicon repair skills trade latency/energy for numerical correctness).

| Scenario | Device | Baseline energy (J) | Final energy (J) | Energy saved | Baseline lat (s) | Final lat (s) |
|----------|--------|--------------------:|-----------------:|-------------:|-----------------:|--------------:|
| H100 + LTX-2.3 + 720p | H100 | 7901 | 2569 | 67.5% | 14.56 | 4.74 |
| RTX 5090 + LTX-2.3 + NVFP4 | RTX5090 | 3949 | 828 | 79.0% | 8.87 | 1.95 |
| RTX 4090 + distill + LTX-2.3 + 480p | RTX4090 | 3930 | 825 | 79.0% | 11.27 | 2.48 |
| M4 Max + LTX-2.3 + 480p | M4_Max | 24959 | 11525 | 53.8% | 68.85 | 28.57 |
| Jetson Thor + LTX-2.3 distill | Jetson_Thor_T5000 | 1440 | 543 | 62.3% | 14.40 | 5.12 |
| Ascend 910B + LTX-2.3 + INT8 | Ascend_910B | 2789 | 1153 | 58.7% | 11.62 | 4.32 |

## Focus Strategy (Direct Simulation)

Each scenario's headline acceleration strategy, simulated directly (independent of the governance auto-selection). This shows the strategy's raw numbers -- governance may reject it when it violates the quality floor or SLO.

| Scenario | Focus strategy | Applicable | Lat (s) | Energy (J) | Memory (GB) | Quality | vs baseline lat |
|----------|----------------|:----------:|--------:|-----------:|------------:|--------:|----------------:|
| H100 + LTX-2.3 + 720p | 720p high-res datacenter | - | - | - | - | - | - |
| RTX 5090 + LTX-2.3 + NVFP4 | NVFP4 (Blackwell FP4) | yes | 3.14 | 979 | 0.66 | 78.00 | 64.6% |
| RTX 4090 + distill + LTX-2.3 + 480p | step distillation (4-step) | yes | 2.03 | 709 | 5.24 | 81.00 | 82.0% |
| M4 Max + LTX-2.3 + 480p | Apple Silicon (repair) | - | - | - | - | - | - |
| Jetson Thor + LTX-2.3 distill | edge distillation | yes | 2.60 | 260 | 5.24 | 81.00 | 82.0% |
| Ascend 910B + LTX-2.3 + INT8 | INT8 (edge NPU) | yes | 6.02 | 1229 | 1.31 | 80.50 | 48.2% |

## Repair Recommendations

NumericalProbe-driven repair skills recommended per scenario (AdaLN/GELU fp32 guards on low-precision backends; boundary-block bf16 on int8-only devices).

- **RTX 5090 + LTX-2.3 + NVFP4** (RTX5090): adaln_fp32
    - adaln_fp32: speedup 0.900, quality_delta 2.50 -- NumericalProbe reported divergence on adln_modulate for RTX 5090 @ bf16 (simulated).
- **RTX 4090 + distill + LTX-2.3 + 480p** (RTX4090): adaln_fp32
    - adaln_fp32: speedup 0.900, quality_delta 2.50 -- NumericalProbe reported divergence on adln_modulate for RTX 4090 @ bf16 (simulated).
- **M4 Max + LTX-2.3 + 480p** (M4_Max): gelu_fp32, adaln_fp32
    - gelu_fp32: speedup 0.920, quality_delta 2.00 -- NumericalProbe reported divergence on adln_modulate, gelu_tanh for M4 Max @ bf16 (simulated).
    - adaln_fp32: speedup 0.900, quality_delta 2.50 -- NumericalProbe reported divergence on adln_modulate, gelu_tanh for M4 Max @ bf16 (simulated).
- **Jetson Thor + LTX-2.3 distill** (Jetson_Thor_T5000): adaln_fp32
    - adaln_fp32: speedup 0.900, quality_delta 2.50 -- NumericalProbe reported divergence on adln_modulate for Jetson AGX Thor T5000 @ bf16 (simulated).
- **Ascend 910B + LTX-2.3 + INT8** (Ascend_910B): adaln_fp32, boundary_block_bf16
    - adaln_fp32: speedup 0.900, quality_delta 2.50 -- NumericalProbe reported divergence on adln_modulate for Ascend 910B @ bf16 (simulated).
    - boundary_block_bf16: speedup 1.000, quality_delta 0.00 -- R4: int8-only device (Ascend 910B) deploying fp8-trained weights (LTX-2.

## Top-5 Ranked Alternatives

### H100 + LTX-2.3 + 720p

| Combo | Lat (s) | Energy (J) | Memory (GB) | Quality | Pareto | Feasible |
|-------|--------:|-----------:|------------:|--------:|--------|:--------:|
| step_distill:8step | 4.74 | 2569 | 8.84 | 83.00 | efficient | yes |
| teacache:thr0.2 | 5.15 | 2792 | 8.84 | 83.23 | efficient | yes |
| teacache:thr0.1 | 7.45 | 4042 | 8.84 | 83.70 | efficient | yes |
| flash_attention | 10.32 | 5597 | 8.84 | 84.00 | efficient | yes |
| teacache+step_distill | 1.34 | 729 | 8.84 | 80.70 | feasible_lowq | no |

### RTX 5090 + LTX-2.3 + NVFP4

| Combo | Lat (s) | Energy (J) | Memory (GB) | Quality | Pareto | Feasible |
|-------|--------:|-----------:|------------:|--------:|--------|:--------:|
| teacache+sage_attention | 1.78 | 715 | 5.24 | 83.60 | efficient | yes |
| sage_attention:v3 | 2.15 | 669 | 5.24 | 82.00 | efficient | yes |
| sage_attention+compile_graph | 2.47 | 990 | 5.24 | 83.90 | efficient | yes |
| compile_graph:torch_compile | 6.29 | 2798 | 5.24 | 84.00 | efficient | yes |
| flash_attention | 6.29 | 2798 | 5.24 | 84.00 | efficient | yes |

### RTX 4090 + distill + LTX-2.3 + 480p

| Combo | Lat (s) | Energy (J) | Memory (GB) | Quality | Pareto | Feasible |
|-------|--------:|-----------:|------------:|--------:|--------|:--------:|
| teacache+sage_attention | 2.27 | 711 | 5.24 | 83.60 | efficient | yes |
| sage_attention+compile_graph | 3.14 | 985 | 5.24 | 83.90 | efficient | yes |
| compile_graph:torch_compile | 7.98 | 2784 | 5.24 | 84.00 | efficient | yes |
| flash_attention | 7.98 | 2784 | 5.24 | 84.00 | efficient | yes |
| step_distill+teacache+sage_attention | 0.40 | 124 | 5.24 | 80.50 | feasible_lowq | no |

### M4 Max + LTX-2.3 + 480p

| Combo | Lat (s) | Energy (J) | Memory (GB) | Quality | Pareto | Feasible |
|-------|--------:|-----------:|------------:|--------:|--------|:--------:|
| teacache:thr0.2 | 24.33 | 8820 | 5.24 | 83.23 | efficient | yes |
| teacache:thr0.1 | 35.23 | 12769 | 5.24 | 83.70 | efficient | yes |
| step_distill:8step | 22.39 | 8115 | 5.24 | 83.00 | efficient | yes |
| mlx_sdpa | 51.73 | 18751 | 4.45 | 84.00 | efficient | yes |
| teacache+step_distill | 6.35 | 2303 | 5.24 | 80.70 | feasible_lowq | no |

### Jetson Thor + LTX-2.3 distill

| Combo | Lat (s) | Energy (J) | Memory (GB) | Quality | Pareto | Feasible |
|-------|--------:|-----------:|------------:|--------:|--------|:--------:|
| step_distill:8step | 4.68 | 468 | 5.24 | 83.00 | efficient | yes |
| teacache:thr0.2 | 5.09 | 509 | 5.24 | 83.23 | efficient | yes |
| teacache+compile_graph | 5.22 | 522 | 5.24 | 83.70 | efficient | yes |
| compile_graph:torch_compile | 10.20 | 1020 | 5.24 | 84.00 | efficient | yes |
| quantization+compile_graph | 10.20 | 1020 | 5.24 | 84.00 | efficient | yes |

### Ascend 910B + LTX-2.3 + INT8

| Combo | Lat (s) | Energy (J) | Memory (GB) | Quality | Pareto | Feasible |
|-------|--------:|-----------:|------------:|--------:|--------|:--------:|
| step_distill:8step | 3.78 | 907 | 5.24 | 83.00 | efficient | yes |
| teacache:thr0.2 | 4.11 | 985 | 5.24 | 83.23 | efficient | yes |
| teacache+compile_graph | 4.21 | 1011 | 5.24 | 83.70 | efficient | yes |
| compile_graph:torch_compile | 8.23 | 1976 | 5.24 | 84.00 | efficient | yes |
| quantization+compile_graph | 8.23 | 1976 | 5.24 | 84.00 | efficient | yes |

---
Generated by scripts/generate_report.py from test_results/results.json (produced by scripts/run_all_scenarios.py). All numbers are roofline + energy-model simulations; no GPU or network was used.
