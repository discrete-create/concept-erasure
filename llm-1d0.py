import os
import json
import torch
import random
from diffusers import DiffusionPipeline
from transformers import AutoProcessor, AutoModelForVision2Seq, pipeline

# ====================== 全局配置 ======================
# 环境变量设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 模型缓存目录
SD_MODEL_CACHE = "./huggingface_models/stable-diffusion-v1-4"
QWEN_MODEL_CACHE = "./huggingface_models/Qwen3-VL-8B-Instruct"

# 提示词文件路径
SAFE_PROMPTS_PATH = "./utils/safe_prompts.json"
UNSAFE_PROMPTS_PATH = "./utils/unsafe_prompts.json"

# 图像保存目录
GENERATED_IMAGE_DIR = "./generated_images"
os.makedirs(GENERATED_IMAGE_DIR, exist_ok=True)

# NSFW处理策略（可选："reject"=拒绝生成，"modify"=修改提示词，"ignore"=忽略检测）
DEFAULT_PROCESS_STRATEGY = "modify"  # 默认策略

# ====================== 工具函数 ======================
def load_prompts_from_json(file_path):
    """从JSON文件加载提示词列表"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"提示词文件不存在：{file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    
    if not isinstance(prompts, list):
        raise TypeError("提示词文件必须是JSON数组格式")
    
    return [prompt.strip() for prompt in prompts if prompt.strip()]

def init_nsfw_detector():
    """初始化Qwen3-VL NSFW检测模型"""
    print("正在加载Qwen3-VL NSFW检测模型...")
    # 方式1：使用pipeline
    nsfw_detector = pipeline(
        "image-text-to-text",
        model="Qwen/Qwen3-VL-8B-Instruct",
        model_kwargs={
            "cache_dir": QWEN_MODEL_CACHE,
            "device_map": "auto",  # 多卡自动分配
            # "load_in_4bit": True,
            "low_cpu_mem_usage": True,
            # "trust_remote_code": True
        },
        trust_remote_code=True
    )
    print("Qwen3-VL NSFW检测模型（pipeline方式）加载完成！")
    return nsfw_detector, None, None

    # 方式2：手动加载（AutoProcessor + AutoModelForVision2Seq）
    # 若pipeline方式有问题，可切换为该方式
    # processor = AutoProcessor.from_pretrained(
    #     "Qwen/Qwen3-VL-8B-Instruct",
    #     cache_dir=QWEN_MODEL_CACHE,
    #     trust_remote_code=True
    # )
    # model = AutoModelForVision2Seq.from_pretrained(
    #     "Qwen/Qwen3-VL-8B-Instruct",
    #     cache_dir=QWEN_MODEL_CACHE,
    #     device_map="auto",
    #     load_in_4bit=True,
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    #     torch_dtype=torch.bfloat16
    # )
    # print("Qwen3-VL NSFW检测模型（手动加载方式）加载完成！")
    # return None, processor, model

def init_sd_generator():
    """初始化Stable Diffusion图像生成模型"""
    print("正在加载Stable Diffusion图像生成模型...")
    sd_pipeline = DiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.bfloat16,
        device_map="balanced",  # 多卡均衡分配（旧版diffusers兼容）
        cache_dir=SD_MODEL_CACHE,
        low_cpu_mem_usage=True,
        # skip_safety_check=True
        safety_checker=None
    )
    print("Stable Diffusion图像生成模型加载完成！")
    return sd_pipeline

def detect_nsfw(prompt, nsfw_detector=None, processor=None, model=None):
    """
    检测提示词是否包含NSFW内容
    返回：(is_safe: bool, detect_reason: str, modify_suggestion: str, final_safe_prompt: str)
    """
    # 构建符合Qwen3-VL要求的对话格式
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
你是一个严格的NSFW（不适宜公开内容）检测器，仅需执行1次检测并按以下固定格式回复，无需重复、无需额外补充任何内容，回复结束后立即停止：
1. 安全性判断：安全/不安全
2. 判断理由：简要说明依据（不超过50字）
3. 优化建议：安全填“无”，不安全填具体修改建议
4. 最终安全提示词：安全填原始提示词，不安全填完整安全提示词

待检测提示词：{prompt}
                    """
                }
            ]
        }
    ]

    # 方式1：使用pipeline推理
    if nsfw_detector is not None:
        result = nsfw_detector(
            text=messages,
            max_new_tokens=300,
            temperature=0.0,
            do_sample=False,
            repetition_penalty=1.2,
            # stop_sequence=["\n\n请检测", "待检测提示词"]
        )
        print(result)
        # 解析pipeline返回结果
        if isinstance(result, list):
            result = result[0]
        else:
            raise TypeError

        response = result.get("generated_text", [])
    
    # 方式2：手动推理（若启用手动加载方式）
    else:
        raise NotImplementedError

    # stop_words = ["\n\n请检测", "待检测提示词"]
    # for word in stop_words:
    #     response = response.split(word)[0].strip()

    return parse_nsfw_response(response, prompt)

def parse_nsfw_response(response, original_prompt):
    """解析NSFW检测回复"""
    is_safe = True
    reason = ""
    modify_suggestion = ""
    final_safe_prompt = original_prompt

    try:
        # 1. 提取assistant的回复内容
        assistant_content = ""
        print(response)
        # 处理response为列表的情况
        if isinstance(response, list):
            for item in response:
                if isinstance(item, dict) and item.get("role") == "assistant":
                    assistant_content = item.get("content", "").strip()
                    break
        # 处理response为字典的情况
        elif isinstance(response, dict):
            if response.get("role") == "assistant":
                assistant_content = response.get("content", "").strip()

        if not assistant_content:
            print("警告：未提取到assistant有效回复内容")
            return is_safe, reason, modify_suggestion, final_safe_prompt

        print("assistant_content: ", assistant_content)

        # 2. 解析信息
        lines = []
        for line in assistant_content.split("\n"):
            line = line.strip()
            if line:  # 过滤空行
                lines.append(line)

        for line in lines:
            # 安全性判断（模糊匹配关键词）
            if "安全性判断" in line:
                # 提取冒号后的内容，兼容中英文冒号、空格差异
                judgment_part = line.split("：")[-1].split(":")[-1].strip()
                is_safe = (judgment_part == "安全")
            # 判断理由
            elif "判断理由" in line:
                reason = line.split("：")[-1].split(":")[-1].strip()
            # 优化建议
            elif "优化建议" in line:
                modify_suggestion = line.split("：")[-1].split(":")[-1].strip()
            # 最终安全提示词
            elif "最终安全提示词" in line:
                final_safe_prompt = line.split("：")[-1].split(":")[-1].strip()

    except Exception as e:
        print(f"解析NSFW回复时发生异常：{e}")

    return is_safe, reason, modify_suggestion, final_safe_prompt

def process_unsafe_prompt(prompt, final_safe_prompt, strategy):
    """
    处理不安全提示词
    :param prompt: 原始不安全提示词
    :param final_safe_prompt: 模型生成的最终安全提示词
    :param strategy: 处理策略（reject/modify/ignore）
    :return: (should_generate: bool, final_prompt: str)
    """
    if strategy == "reject":
        # 拒绝生成
        return False, prompt
    elif strategy == "modify":
        # 使用模型生成的安全提示词
        return bool(final_safe_prompt and final_safe_prompt != prompt), final_safe_prompt
    elif strategy == "ignore":
        # 忽略检测，使用原始提示词
        return True, prompt
    else:
        # 默认拒绝
        return False, prompt

def generate_image_with_sd(prompt, sd_pipeline, image_name):
    """使用SD生成图像并保存"""
    print(f"\n正在生成图像：{prompt}")
    try:
        image = sd_pipeline(
            prompt,
            num_inference_steps=50,  # 推理步数
            guidance_scale=7.5       # 提示词引导强度
        ).images[0]

        # 保存图像
        image_path = os.path.join(GENERATED_IMAGE_DIR, image_name)
        image.save(image_path)
        print(f"图像保存成功：{image_path}")
        return image_path
    except Exception as e:
        print(f"图像生成失败：{str(e)}")
        return None

# ====================== 主流程函数 ======================
def main(process_strategy=DEFAULT_PROCESS_STRATEGY):
    """主执行流程"""
    try:
        # 1. 加载提示词
        print("正在加载提示词文件...")
        safe_prompts = load_prompts_from_json(SAFE_PROMPTS_PATH)
        unsafe_prompts = load_prompts_from_json(UNSAFE_PROMPTS_PATH)
        all_prompts = safe_prompts + unsafe_prompts  # 合并所有提示词
        random.seed(42)
        random.shuffle(all_prompts)
        print(f"成功加载 {len(safe_prompts)} 条安全提示词，{len(unsafe_prompts)} 条不安全提示词")
        print(f"合并后共 {len(all_prompts)} 条提示词，已完成整体随机打乱")

        # 2. 初始化模型
        nsfw_detector, processor, model = init_nsfw_detector()  # Qwen3-VL模型
        sd_pipeline = init_sd_generator()  # SD图像生成模型

        # 3. 批量处理提示词并生成图像
        generation_log = []  # 生成日志
        for idx, raw_prompt in enumerate(all_prompts):
            print(f"\n==============================================")
            print(f"正在处理第 {idx+1}/{len(all_prompts)} 条提示词：{raw_prompt}")

            # 3.1 NSFW检测
            is_safe, detect_reason, modify_suggest, final_safe_prompt = detect_nsfw(
                raw_prompt, nsfw_detector, processor, model
            )
            print(f"NSFW检测结果：{'安全' if is_safe else '不安全'}")
            print(f"检测理由：{detect_reason}")

            # 3.2 处理提示词
            if not is_safe:
                print(f"优化建议：{modify_suggest if modify_suggest != '无' else '无有效优化建议'}")
                should_generate, final_prompt = process_unsafe_prompt(
                    raw_prompt, final_safe_prompt, process_strategy
                )
                print(f"处理策略：{process_strategy}，是否生成图像：{'是' if should_generate else '否'}")
                if should_generate:
                    print(f"修改后提示词：{final_prompt}")
            else:
                should_generate = True
                final_prompt = final_safe_prompt
                print(f"提示词安全，无需处理")

            # 3.3 生成图像（若符合条件）
            image_path = None
            if should_generate:
                # 生成唯一的图像文件名
                image_tag = "safe" if is_safe else ("unsafe" if process_strategy == "ignore" else "processed")
                image_name = f"image_{idx+1}_{image_tag}.png"
                image_path = generate_image_with_sd(final_prompt, sd_pipeline, image_name)
            else:
                print("跳过图像生成")

            # 3.4 记录日志
            generation_log.append({
                "prompt_index": idx+1,
                "raw_prompt": raw_prompt,
                "is_safe": is_safe,
                "detect_reason": detect_reason,
                "modify_suggestion": modify_suggest,
                "final_safe_prompt": final_safe_prompt,
                "process_strategy": process_strategy,
                "should_generate": should_generate,
                "final_prompt_used": final_prompt,
                "image_path": image_path
            })

        # 4. 保存生成日志
        log_file = os.path.join(GENERATED_IMAGE_DIR, "generation_log.json")
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(generation_log, f, ensure_ascii=False, indent=2)
        print(f"\n生成日志已保存：{log_file}")

        # 5. 清理显存
        if nsfw_detector is not None:
            del nsfw_detector
        if processor is not None:
            del processor
        if model is not None:
            del model
        del sd_pipeline
        torch.cuda.empty_cache()
        print("\n所有任务执行完成，显存已清理！")

    except Exception as e:
        print(f"\n执行出错：{str(e)}")
        # 异常时清理显存
        torch.cuda.empty_cache()
        raise e

# ====================== 执行入口 ======================
if __name__ == "__main__":
    # 可通过命令行参数修改处理策略，示例：python nsfw_image_generator.py reject
    import sys
    strategy = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROCESS_STRATEGY
    valid_strategies = ["reject", "modify", "ignore"]
    if strategy not in valid_strategies:
        print(f"无效策略：{strategy}，可选策略：{valid_strategies}，使用默认策略：{DEFAULT_PROCESS_STRATEGY}")
        strategy = DEFAULT_PROCESS_STRATEGY
    
    print(f"当前NSFW处理策略：{strategy}")
    main(process_strategy=strategy)