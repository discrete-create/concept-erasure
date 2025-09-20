from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel,DDIMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
import numpy as np
import torch
from diffusers import UNet2DConditionModel
from transformers import AutoTokenizer, PretrainedConfig
from utils.config_utils import import_models_for_cfr
from utils.image_part import latent2image, show_and_save_image,generate_imag
import os

Model_path='/root/autodl-tmp/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# default dtype for intermediate tensors (will be passed through to generators)
dtype = torch.float16
print(f"Using device: {device}, dtype: {dtype}")
tokenizer, text_encoder, unet = import_models_for_cfr(Model_path, device, dtype=dtype)
vae = AutoencoderKL.from_pretrained(Model_path, subfolder="vae").to(device)
scheduler = DDIMScheduler.from_pretrained(Model_path, subfolder="scheduler")
prompts = "a photo of an astronaut riding a horse on mars"
output_dir="./output"
os.makedirs(output_dir, exist_ok=True)
num_inference_steps=500
guidance_scale=7.5
batch_size = 1
# create latents with correct dtype and device
latents = torch.randn((batch_size, unet.in_channels, 64, 64), dtype=dtype, device=device)

# run denoising -> returns latents
latents_out = generate_imag(
    latents,
    prompts,
    tokenizer,
    text_encoder,
    unet,
    scheduler,
    guidance_scale=guidance_scale,
    device=device,
    num_inference_steps=num_inference_steps,
    eta=0.0,
    dtype=dtype,
)

# decode latents to images (numpy uint8) and save/show
decoded = latent2image(vae, latents_out, return_type="np")
# decoded shape: (B, H, W, C)
print(f"Decoded {decoded.shape[0]} images.")
for i, image in enumerate(decoded):
    save_path = os.path.join(output_dir, f"generated_image_{i}.png")
    show_and_save_image(image, save_path=save_path, title=f"Generated Image {i}")
    print(f"Image {i} saved to {save_path}")