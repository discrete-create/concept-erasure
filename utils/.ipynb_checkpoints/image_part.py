from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel,DDIMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
import numpy as np
import torch
import numpy as np

def latent2image(vae, latents, return_type=('np', 'pt')):
    """
    将latent向量解码为图像，可返回numpy和torch两种形式
    Args:
        vae: 训练好的VAE模型
        latents: 输入latent tensor (B, C, H, W)
        return_type: 'np', 'pt' 或 ('np','pt')，支持同时返回
    Returns:
        dict，包含 'np' 和/或 'pt' 的图像
    """
    # 还原缩放
    latents = 1 / 0.18215 * latents.detach()

    # 解码
    image = vae.decode(latents)["sample"]  # (B, C, H, W)

    # 归一化到 [0,1]
    image = (image / 2 + 0.5).clamp(0, 1)

    results = {}

    if isinstance(return_type, str):
        return_type = (return_type,)

    if "pt" in return_type:
        results["pt"] = image  # (B, C, H, W)，范围[0,1]

    if "np" in return_type:
        # 转 numpy
        np_img = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        np_img = (np_img * 255).astype(np.uint8)
        results["np"] = np_img  # (B, H, W, C)

    # 如果只要一种，直接返回；否则返回字典
    if len(results) == 1:
        return list(results.values())[0]
    return results

import os
import matplotlib.pyplot as plt
import torch
import numpy as np

def show_and_save_image(image, save_path=None, title=None):
    """
    可视化并保存图像
    Args:
        image: torch.Tensor (C,H,W) or (B,C,H,W) or np.ndarray (H,W,C) or (B,H,W,C)
        save_path: 保存路径 (str)，例如 "output.png"，默认不保存
        title: 显示在图像上方的标题
    """
    # 处理 torch.Tensor
    if isinstance(image, torch.Tensor):
        # 去掉 batch 维
        if image.dim() == 4:  
            image = image[0]
        # (C,H,W) -> (H,W,C)
        image = image.detach().cpu().permute(1, 2, 0).numpy()
        # 转到 [0,255]
        image = (image * 255).astype(np.uint8)

    # 处理 np.ndarray
    elif isinstance(image, np.ndarray):
        if image.ndim == 4:  # batch
            image = image[0]
        # 确保类型正确
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

    else:
        raise TypeError("输入必须是 torch.Tensor 或 np.ndarray")

    # 可视化
    plt.imshow(image)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()

    # 保存
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.imsave(save_path, image)
        print(f"图片已保存到 {save_path}")

def generate_step(
    scheduler,
    noise_pred :torch.FloatTensor,
    timestep :int,
    sample :torch.FloatTensor,
    eta :float = 0.0
):
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    prev_timestep = max(0, prev_timestep)
    prev_alpha = scheduler.alphas_cumprod[prev_timestep]
    alpha = scheduler.alphas_cumprod[timestep]
    beta = 1 - alpha
    pred_dir = (1 - prev_alpha)**0.5 * noise_pred
    pred_X0 = (sample - beta**0.5 * noise_pred) / alpha**0.5
    pred_X=prev = prev_alpha**0.5 * pred_X0 + pred_dir
    noise = torch.randn_like(sample)
    if eta > 0:
        sigma = eta * ((1 - alpha / prev_alpha) * (1 - prev_alpha) / (1 - alpha))**0.5
        pred_X = pred_X + sigma * noise
    return pred_X, pred_X0

def generate_imag(latents,prompts,tokenizer,text_encoder,unet,scheduler,device,guidance_scale=7.5,num_inference_steps=50,eta=0.0):
    text_input = tokenizer(
        prompts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(scheduler.timesteps):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale* (noise_pred_text - noise_pred_uncond)

        latents, _ = generate_step(scheduler, noise_pred, t, latents, eta)

    return latents
