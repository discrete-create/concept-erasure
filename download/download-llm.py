"""
download qwen3-vl-8b
"""
from transformers import pipeline

model_cache_dir = "./huggingface_models/Qwen3-VL-8B-Instruct"

pipe = pipeline("image-text-to-text", model="Qwen/Qwen3-VL-8B-Instruct", model_kwargs={"cache_dir": model_cache_dir, "device_map": "auto"})
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "download/candy.jpg"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
result = pipe(text=messages)
print(result)