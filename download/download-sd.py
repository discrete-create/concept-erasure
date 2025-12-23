"""
download sd model
"""
import torch
from diffusers import DiffusionPipeline

model_cache_dir = "./huggingface_models/stable-diffusion-v1-4"

pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.bfloat16, device_map="balanced", cache_dir=model_cache_dir)

prompt = "A high tech solarpunk utopia in the Amazon rainforest"
image = pipe(prompt).images[0]
image.save("download/solarpunk_utopia.png")

