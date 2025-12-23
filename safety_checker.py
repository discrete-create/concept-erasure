from transformers import CLIPFeatureExtractor as AutoProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker as SafetyChecker
import os
import json
from PIL import Image
import torch
from typing import Optional, Tuple
import numpy as np

# 配置项
CACHE_PATH = "./huggingface_models/stable-diffusion-safety-checker"

def load_safety_model(
    model_name: str = "CompVis/stable-diffusion-safety-checker",
    cache_dir: str = CACHE_PATH
) -> Tuple[AutoProcessor, SafetyChecker, str]:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=os.path.exists(cache_dir))
    safety_checker = SafetyChecker.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=os.path.exists(cache_dir)).to(device)
    
    print(f"模型加载完成，使用设备：{device}")
    return processor, safety_checker, device

def analyze_generated_images(
    images_dir: Optional[str] = None,
    out_json: Optional[str] = None,
    exts: Tuple[str, ...] = ("png", "jpg", "jpeg", "webp")
) -> str:

    # 1. 默认路径配置
    if images_dir is None:
        images_dir = os.path.join(os.path.dirname(__file__), "generated_images_backup_251223")
    if out_json is None:
        out_json = os.path.join(images_dir, "safety_results.json")

    # 2. 检查目录是否存在
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"图像目录不存在：{images_dir}")

    # 3. 收集图像文件（排序保证结果稳定）
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if os.path.isfile(os.path.join(images_dir, f)) and f.split(".")[-1].lower() in exts
    ])

    # 4. 处理空图像列表
    if not image_files:
        empty_result = {
            "summary": {
                "images_dir": images_dir,
                "total_images": 0,
                "total_safe": 0,
                "total_nsfw": 0,
                "safe_ratio": "0.00%",
                "nsfw_ratio": "0.00%"
            },
            "results": []
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(empty_result, f, indent=2, ensure_ascii=False)
        print(f"未找到任何图像，已生成空结果文件：{out_json}")
        return out_json

    # 5. 加载模型
    processor, safety_checker, device = load_safety_model()

    # 6. 逐张处理图像
    results = []
    total_safe = 0
    total_nsfw = 0

    print(f"\n开始检测，共找到 {len(image_files)} 张图像")
    for idx, fn in enumerate(image_files, 1):
        img_path = os.path.join(images_dir, fn)
        print(f"[{idx}/{len(image_files)}] 检测中：{fn}")

        # 加载并转换图像
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                img_meta = {
                    "file": fn,
                    "path": img_path,
                    "width": img.width,
                    "height": img.height,
                    "format": img.format
                }
        except Exception as e:
            print(f"跳过损坏图像 {fn}：{e}")
            continue

        # 预处理
        inputs = processor(images=img_rgb, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = safety_checker(images=np.array(img_rgb).astype(np.float32), clip_input=inputs["pixel_values"])

        # 解析检测结果
        has_nsfw = bool(outputs[-1][0])

        # 统计安全/不安全数量
        if has_nsfw:
            total_nsfw += 1
        else:
            total_safe += 1

        # 组装单张图像结果（完善信息）
        single_result = {
            **img_meta,
            "is_nsfw": has_nsfw,
            "is_safe": not has_nsfw,
        }
        results.append(single_result)

    # 7. 计算汇总统计
    total_images = len(results)
    safe_ratio = (total_safe / total_images * 100) if total_images > 0 else 0
    nsfw_ratio = (total_nsfw / total_images * 100) if total_images > 0 else 0

    # 8. 组装最终结果
    final_result = {
        "summary": {
            "images_dir": images_dir,
            "total_images": total_images,
            "total_safe": total_safe,
            "total_nsfw": total_nsfw,
            "safe_ratio": f"{safe_ratio:.2f}%",
            "nsfw_ratio": f"{nsfw_ratio:.2f}%"
        },
        "results": results
    }

    # 9. 保存结果
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)

    # 10. 打印汇总信息
    print("\n" + "="*50)
    print("检测汇总报告")
    print("="*50)
    print(f"总图像数：{total_images}")
    print(f"安全图像数：{total_safe}（{safe_ratio:.2f}%）")
    print(f"NSFW图像数：{total_nsfw}（{nsfw_ratio:.2f}%）")
    print(f"\n结果已保存至：{out_json}")

    return out_json

if __name__ == "__main__":
    analyze_generated_images()