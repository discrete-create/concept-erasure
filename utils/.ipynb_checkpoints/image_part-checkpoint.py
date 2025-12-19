from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel,DDIMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
import numpy as np
import torch
import numpy as np
from scipy.stats import truncnorm, expon
import torch.nn.functional as F

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

    # move/cast latents to the same device and dtype as the VAE parameters to avoid
    # mixed-type operations (e.g. conv bias float vs input half) which raise runtime errors
    vae_device = next(vae.parameters()).device if any(True for _ in vae.parameters()) else torch.device("cpu")
    # prefer the dtype of the VAE parameters (usually float32)
    try:
        vae_dtype = next(vae.parameters()).dtype
    except StopIteration:
        vae_dtype = latents.dtype

    latents = latents.to(device=vae_device, dtype=vae_dtype)

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
    noise_pred: torch.Tensor,
    timestep: int,
    sample: torch.Tensor,
    eta: float = 0.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
):
    """
    Single denoising step that ensures intermediate tensors are created with `dtype`.
    """
    if device is None:
        device = sample.device

    # compute previous timestep index safely
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    prev_timestep = max(0, prev_timestep)

    # make sure alphas are tensors with correct dtype/device
    # scheduler.alphas_cumprod may be a numpy array or torch tensor
    alphas = torch.as_tensor(scheduler.alphas_cumprod, dtype=dtype, device=device)
    prev_alpha = alphas[prev_timestep]
    alpha = alphas[timestep]

    beta = 1 - alpha

    pred_dir = (1 - prev_alpha).sqrt() * noise_pred
    pred_X0 = (sample - beta.sqrt() * noise_pred) / alpha.sqrt()
    pred_X = prev_alpha.sqrt() * pred_X0 + pred_dir

    # create noise in requested dtype/device
    noise = torch.randn_like(sample, dtype=dtype, device=device)
    if eta > 0:
        # compute sigma in dtype/device
        # use float64 for intermediate math stability then cast
        sigma = eta * (((1 - (alpha / prev_alpha)) * (1 - prev_alpha) / (1 - alpha)).to(dtype)).sqrt()
        pred_X = pred_X + sigma * noise

    return pred_X, pred_X0

def generate_imag(latents, prompts, tokenizer, text_encoder, unet, scheduler, device,
                  guidance_scale=7.5, num_inference_steps=20, eta=0.0, dtype=torch.float16):
    """
    Generate latents using classifier-free guidance.

    Args:
        latents: initial latent tensor of shape (B, C, H, W)
        prompts: either a single string, or list of `B` strings (conditional prompts)
        tokenizer, text_encoder, unet, scheduler: models/objects
        device: torch device
        guidance_scale: classifier-free guidance scale
        num_inference_steps, eta, dtype: scheduler / numeric params

    Returns:
        latents tensor after denoising (shape (B, C, H, W))
    """
    # normalize prompts to list
    if isinstance(prompts, str):
        prompts = [prompts]

    # batch size from latents
    batch_size = latents.shape[0]

    # ensure number of conditional prompts matches batch_size
    if len(prompts) != batch_size:
        # if user passed a single conditional prompt but batch_size > 1, broadcast
        if len(prompts) == 1:
            prompts = prompts * batch_size
        else:
            raise ValueError(f"Number of prompts ({len(prompts)}) must equal batch size ({batch_size}) or be 1.")

    # tokenize conditional prompts
    cond_input = tokenizer(
        prompts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    # tokenize unconditional (empty) prompts for classifier-free guidance
    uncond_input = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    # move input ids and get embeddings in correct dtype/device
    cond_ids = cond_input.input_ids.to(device)
    uncond_ids = uncond_input.input_ids.to(device)

    with torch.no_grad():
        cond_embeddings = text_encoder(cond_ids)[0].to(dtype=dtype, device=device)
        uncond_embeddings = text_encoder(uncond_ids)[0].to(dtype=dtype, device=device)

    # concatenate unconditional and conditional embeddings to shape (2*B, seq, dim)
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings], dim=0)

    # ensure latents are the requested dtype/device
    latents = latents.to(dtype=dtype, device=device)

    scheduler.set_timesteps(num_inference_steps)
    intermediate_latent={}
    for t in scheduler.timesteps:
        # expand latents for classifier-free guidance: (B,...) -> (2*B,...)
        latent_model_input = torch.cat([latents, latents], dim=0).to(dtype=dtype, device=device)

        # some schedulers require scaling the model input
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # split and do guidance in dtype
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents, _ = generate_step(scheduler, noise_pred, t, latents, eta, dtype=dtype, device=device)
        if prompts[0] not in intermediate_latent :
            intermediate_latent[prompts[0]]={}
        if(t<150):
            intermediate_latent[prompts[0]][t]=_
    return latents,intermediate_latent
def control_func(time_step):
    return (time_step-500)/500

def generate_imag_with_b_net(
        latents, prompts, tokenizer, text_encoder, unet, scheduler, device,
        binary_network, sample_length, IMG_SIZE=224,
        guidance_scale=7.5, num_inference_steps=20, eta=0.0, dtype=torch.float16):

    # --- Normalize prompts ---
    if isinstance(prompts, str):
        prompts = [prompts]

    batch_size = latents.shape[0]
    if len(prompts) != batch_size:
        if len(prompts) == 1:
            prompts = prompts * batch_size
        else:
            raise ValueError("Prompts count must match batch size or be 1.")

    # --- Tokenize ---
    cond_ids = tokenizer(prompts, padding="max_length", max_length=77,
                         truncation=True, return_tensors="pt").input_ids.to(device)
    uncond_ids = tokenizer([""] * batch_size, padding="max_length", max_length=77,
                           truncation=True, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        cond_emb = text_encoder(cond_ids)[0].to(dtype=dtype, device=device)
        uncond_emb = text_encoder(uncond_ids)[0].to(dtype=dtype, device=device)

    text_embeddings = torch.cat([uncond_emb, cond_emb], dim=0)

    latents = latents.to(dtype=dtype, device=device)
    scheduler.set_timesteps(num_inference_steps)

    # ============================================================
    #  Begin denoising
    # ============================================================
    for t in tqdm(scheduler.timesteps):
        latent_in = torch.cat([latents, latents], dim=0)
        latent_in = scheduler.scale_model_input(latent_in, t)

        with torch.no_grad():
            noise_pred = unet(latent_in, t, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        # ================================
        # 生成基础 CFG guidance（但不应用）
        # ================================
        base_guidance = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # ================================
        # 更新 latent，得到 imgs（用于 binary_network）
        # ================================
        latents, imgs = generate_step(
            scheduler, base_guidance, t, latents, eta,
            dtype=dtype, device=device
        )

        # 注意：imgs 是四通道：RGB + reserved
        # 严格按照你训练的方式处理
        with torch.no_grad():
            # resize → 训练时的处理
            imgs_resized = torch.nn.functional.interpolate(
                imgs, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False
            )

            reserved = imgs_resized[:, 3:4, :, :]   # (B,1,H,W)
            imgs_rgb = imgs_resized[:, :3, :, :]     # 取 RGB
            imgs_rgb = imgs_rgb + reserved / 3        # 平均加回 RGB

            bnet_in = imgs_rgb     # 现在是 (B,3,H,W)
            bnet_out = binary_network(bnet_in).squeeze()   # (B,)
            pred = (bnet_out > 0).long()    # 0 或 1
            print("pred",pred)
            enable_control = pred

        # ================================
        # 根据 bnet 输出调整 guidance_scale
        # ================================
        final_noise = []
        for i in range(batch_size):
            if enable_control[i].item() == 1:
                # 使用 control_func(t)
                print("trigger")
                gs = guidance_scale #* control_func(t)
            else:
                gs = guidance_scale

            guided = noise_pred_uncond[i] + gs * (noise_pred_text[i] - noise_pred_uncond[i])
            final_noise.append(guided)

        noise_pred = torch.stack(final_noise, dim=0)

        # ================================
        # 用真正的 noise_pred 更新 latent
        # ================================
        latents, imgs = generate_step(
            scheduler, noise_pred, t, latents, eta,
            dtype=dtype, device=device
        )

    return latents
