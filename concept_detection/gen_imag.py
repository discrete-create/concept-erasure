import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel,DDIMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import numpy as np
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

  
  
def step(
        scheduler,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float=0.0,
        verbose=False,
    ):
    """
        predict the sampe the next step in the denoise process.
    """
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
    pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
    x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
    return x_prev, pred_x0
  
def get_x0(scheduler, model_output, x, timestep, eta=0.0, verbose=False):
		alpha_prod_t = scheduler.alphas_cumprod[timestep]
		beta_prod_t = 1 - alpha_prod_t
		pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
		return pred_x0


@torch.no_grad()
def latent2image(vae, latents, return_type='np'):
    latents = 1 / 0.18215 * latents.detach()

    # Ensure latents match the VAE parameter dtype and device to avoid dtype/device mismatch
    try:
        param = next(vae.parameters())
        param_dtype = param.dtype
        param_device = param.device
    except StopIteration:
        # fallback
        param_dtype = latents.dtype
        param_device = latents.device

    latents = latents.to(param_dtype).to(param_device)

    image = vae.decode(latents)['sample']
    if return_type == 'np':
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
    elif return_type == "pt":
        image = (image / 2 + 0.5).clamp(0, 1)
    return image
  
def get_text_embedding(tokenizer,text_encoder,prompts,device):
    
    text_input = tokenizer(
                prompts,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            )
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    
    return text_embeddings
  
  
def gen_image(latents,prompts_emb,tokenizer,text_encoder,unet,scheduler,device,num_inference_steps=20,guidance_scale=7.5):
    batch_size = 1
    
    '''text_input = tokenizer(
                prompts,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )'''
    text_embeddings = prompts_emb
    
    if guidance_scale > 1.:
        uc_text = ""
        unconditional_input = tokenizer(
                    [uc_text] * batch_size,
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt"
                )
    unconditional_embeddings = text_encoder(unconditional_input.input_ids.to(device))[0]
      # text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)
    
    scheduler.set_timesteps(num_inference_steps)
    
    # for i,t in enumerate(tqdm(scheduler.timesteps,desc = "DDIM Sampler")):
    results = []
    for i,t in enumerate(tqdm(scheduler.timesteps)):
        noise_pred_con = unet(latents, t, encoder_hidden_states=text_embeddings).sample
        if guidance_scale > 1.:
            with torch.no_grad():
                noise_pred_uncon = unet(latents, t, encoder_hidden_states=unconditional_embeddings).sample
            noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
        else:
            noise_pred = noise_pred_con
        latents,pred_x0 = step(scheduler, noise_pred, t, latents)
        #results.append([latents,noise_pred, t])
      # break
    return latents
 