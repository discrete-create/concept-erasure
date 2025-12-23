from transformers import CLIPFeatureExtractor as AutoProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker as SafetyChecker
import os
import json
from PIL import Image

cache_path = "./huggingface_models/stable-diffusion-safety-checker"

processor = AutoProcessor.from_pretrained(
    "CompVis/stable-diffusion-safety-checker",
    cache_dir=cache_path
)
model = SafetyChecker.from_pretrained(
    "CompVis/stable-diffusion-safety-checker",
    cache_dir=cache_path
)