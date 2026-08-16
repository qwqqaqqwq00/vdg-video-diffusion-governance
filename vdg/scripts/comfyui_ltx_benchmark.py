#!/usr/bin/env python3
"""Submit LTX-2.3 T2V workflow to ComfyUI API and measure real latency.

Uses GGUF Q8 model (22GB) on M4 Max MPS, with --fp16-unet active
(the exact config that triggers MPS black-video NaN per MPS_BLACK_VIDEO_FIX.md).
"""
import json, time, sys, urllib.request, urllib.error

SERVER = "http://127.0.0.1:8188"

def post_prompt(workflow):
    data = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(f"{SERVER}/prompt", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

def get_history(prompt_id):
    try:
        with urllib.request.urlopen(f"{SERVER}/history/{prompt_id}", timeout=10) as resp:
            return json.loads(resp.read())
    except:
        return {}

def get_queue():
    try:
        with urllib.request.urlopen(f"{SERVER}/queue", timeout=10) as resp:
            return json.loads(resp.read())
    except:
        return {}

# Minimal LTX-2.3 T2V workflow (API format)
# Node IDs as strings
workflow = {
    "1": {"class_type": "UnetLoaderGGUF",
          "inputs": {"unet_name": "DasiwaLTX23_goldenLaceV3.gguf"}},
    "2": {"class_type": "VAELoader",
          "inputs": {"vae_name": "LTX23_video_vae_bf16.safetensors"}},
    "3": {"class_type": "LTXVGemmaCLIPModelLoader",
          "inputs": {"gemma_path": "gemma-3-12b/model.safetensors",
                     "ltxv_path": "ltx2310eros_v1.safetensors",
                     "max_length": 256}},
    "4": {"class_type": "CLIPTextEncode",
          "inputs": {"text": "A cinematic shot of a cat playing with a ball of yarn in warm afternoon sunlight, gentle motion, shallow depth of field",
                     "clip": ["3", 0]}},
    "5": {"class_type": "CLIPTextEncode",
          "inputs": {"text": "blurry, low quality, distorted, watermark, static, still frame",
                     "clip": ["3", 0]}},
    "6": {"class_type": "LTXVConditioning",
          "inputs": {"positive": ["4", 0], "negative": ["5", 0], "frame_rate": 25.0}},
    "7": {"class_type": "BasicGuider",
          "inputs": {"model": ["1", 0], "conditioning": ["6", 0]}},
    "8": {"class_type": "KSamplerSelect",
          "inputs": {"sampler_name": "euler"}},
    "9": {"class_type": "BasicScheduler",
          "inputs": {"model": ["1", 0], "scheduler": "normal", "steps": 10, "denoise": 1.0}},
    "10": {"class_type": "RandomNoise",
           "inputs": {"noise_seed": 42}},
    "11": {"class_type": "LTXVBaseSampler",
           "inputs": {"model": ["1", 0], "vae": ["2", 0],
                      "width": 768, "height": 512, "num_frames": 41,
                      "guider": ["7", 0], "sampler": ["8", 0],
                      "sigmas": ["9", 0], "noise": ["10", 0]}},
    "12": {"class_type": "VAEDecode",
           "inputs": {"samples": ["11", 0], "vae": ["2", 0]}},
    "13": {"class_type": "SaveAnimatedWEBP",
           "inputs": {"images": ["12", 0], "filename_prefix": "vdg_ltx_test",
                      "fps": 25.0, "lossless": False, "quality": 80, "method": "default"}},
}

print("=== VDG LTX-2.3 端到端实测 ===")
print(f"模型: DasiwaLTX23_goldenLaceV3.gguf (22GB Q8, ~19B params)")
print(f"VAE: LTX23_video_vae_bf16")
print(f"文本编码器: Gemma-3-12B fp8 + LTX embeddings connector")
print(f"分辨率: 768x512, 帧数: 41, 步数: 10, 采样器: euler")
print(f"设备: M4 Max MPS (--fp16-unet active)")
print()

# Submit
print("提交工作流到 ComfyUI API...")
t0 = time.time()
try:
    result = post_prompt(workflow)
except urllib.error.HTTPError as e:
    print(f"提交失败: HTTP {e.code}")
    err = e.read().decode()
    print(f"错误: {err[:2000]}")
    sys.exit(1)
except Exception as e:
    print(f"提交异常: {e}")
    sys.exit(1)

prompt_id = result.get("prompt_id", "")
print(f"prompt_id: {prompt_id}")
if not prompt_id:
    print(f"响应: {json.dumps(result, indent=2)[:1000]}")
    sys.exit(1)

# Poll for completion
print("\n等待生成完成（轮询 /history）...")
t_submit = time.time()
max_wait = 1800  # 30 min max
last_status = ""
while time.time() - t_submit < max_wait:
    elapsed = time.time() - t0
    history = get_history(prompt_id)
    if prompt_id in history:
        entry = history[prompt_id]
        status = entry.get("status", {})
        completed = status.get("status_str", "")
        if completed:
            t_total = time.time() - t0
            print(f"\n✅ 生成完成! 总耗时: {t_total:.1f}s")
            print(f"状态: {completed}")
            # Get output info
            outputs = entry.get("outputs", {})
            for node_id, node_out in outputs.items():
                print(f"  节点 {node_id} 输出: {json.dumps(node_out)[:200]}")
            # Get execution time from status
            msgs = status.get("messages", [])
            for m in msgs:
                print(f"  消息: {m}")
            print(f"\n=== 实测数据 ===")
            print(f"wall_clock_total_s: {t_total:.1f}")
            print(f"model: LTX_2_3_real (19B Q8 GGUF)")
            print(f"resolution: 768x512")
            print(f"frames: 41")
            print(f"steps: 10")
            print(f"device: M4_Max_MPS")
            break
    # Check queue
    queue = get_queue()
    running = queue.get("queue_running", [])
    pending = queue.get("queue_pending", [])
    status_str = f"running={len(running)} pending={len(pending)} elapsed={elapsed:.0f}s"
    if status_str != last_status:
        print(f"  {status_str}", flush=True)
        last_status = status_str
    time.sleep(5)
else:
    print(f"\n❌ 超时 ({max_wait}s)，生成未完成")
    # Print any error info
    history = get_history(prompt_id)
    if prompt_id in history:
        print(f"历史: {json.dumps(history[prompt_id], indent=2)[:2000]}")
