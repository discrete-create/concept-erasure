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


# steering helper -------------------------------------------------------------





def _create_steering_hook(module_name, steer_bank, strength):
    """
    steer_bank: Tensor [N, C]  (已堆叠的 embedding bank)
    strength: float
    """

    # 预归一化一次
    steer_bank = F.normalize(steer_bank, dim=-1)

    def hook(module, inputs, output):

        # output: [B, C, H, W] or [B, C]
        if output.dim() == 4:
            query = output.mean(dim=[2, 3])  # [B, C]
        else:
            query = output  # [B, C]

        query_norm = F.normalize(query, dim=-1)

        # cosine: [B, C] @ [C, N] -> [B, N]
        sim = torch.matmul(query_norm, steer_bank.T)

        best_idx = sim.argmax(dim=-1)  # [B]

        selected = steer_bank[best_idx]  # [B, C]

        if output.dim() == 4:
            selected = selected.unsqueeze(-1).unsqueeze(-1)

        steer_direction = -selected  # reverse

        return output + strength * steer_direction

    return hook


import torch
import torch.nn.functional as F

def build_steering_schedule(unet, steering_dict):
    """
    将 extract 保存的 steering_dict 转换为:
    {
        timestep_value: {
            module_name: stacked_bank_tensor
        }
    }
    """

    device = next(unet.parameters()).device

    schedule = {}
    name_to_module = dict(unet.named_modules())

    for concept in steering_dict:
        for template in steering_dict[concept]:
            for ts_index in steering_dict[concept][template]:

                for module_name, entry in steering_dict[concept][template][ts_index].items():

                    ts_value = entry["value"]
                    tensor = entry["tensor"]

                    # pooling 到 [C]
                    if tensor.dim() == 4:
                        emb = tensor.mean(dim=[0, 2, 3])
                    elif tensor.dim() == 3:
                        emb = tensor.mean(dim=1).squeeze(0)
                    elif tensor.dim() == 2:
                        emb = tensor.mean(dim=0)
                    else:
                        continue

                    emb = emb.to(device)

                    schedule.setdefault(ts_value, {})
                    schedule[ts_value].setdefault(module_name, [])
                    schedule[ts_value][module_name].append(emb)

    # stack bank
    for t in schedule:
        for module_name in schedule[t]:
            bank = torch.stack(schedule[t][module_name], dim=0)
            schedule[t][module_name] = bank

    return schedule
def register_steering_hooks_for_timestep(unet, schedule, timestep, strength):

    handles = []

    if timestep not in schedule:
        return handles

    name_to_module = dict(unet.named_modules())

    for module_name, steer_bank in schedule[timestep].items():

        if module_name not in name_to_module:
            print(f"[WARNING] {module_name} not found in UNet")
            continue

        module = name_to_module[module_name]

        handle = module.register_forward_hook(
            _create_steering_hook(module_name, steer_bank, strength)
        )

        handles.append(handle)

    return handles


def step_with_steering(
        scheduler,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        layers: list = None,
        steering_embeddings: list = None,
        steering_strength: float = 1.0,
        eta: float = 0.0,
        verbose=False,
    ):
    """Deprecated stub; steering is applied via hooks now.

    This helper retains the old signature for backwards compatibility but does
    *not* modify ``model_output``.  Users should call
    :func:`register_steering_hooks` before generation instead.
    """
    if layers or steering_embeddings:
        print("[step_with_steering] warning: please use register_steering_hooks for steering")
    return step(scheduler, model_output, timestep, x, eta=eta, verbose=verbose)
  
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
  
  
from tqdm import tqdm


def gen_image(latents,
              prompts_emb,
              tokenizer,
              text_encoder,
              unet,
              scheduler,
              device,
              num_inference_steps=50,
              guidance_scale=7.5,
              steering_embeddings=None,
              steering_strength=5.0):

    batch_size = 1
    text_embeddings = prompts_emb

    unconditional_embeddings = None

    if guidance_scale > 1.:
        uc_text = ""
        unconditional_input = tokenizer(
            [uc_text] * batch_size,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        unconditional_embeddings = text_encoder(
            unconditional_input.input_ids.to(device)
        )[0]

    scheduler.set_timesteps(num_inference_steps)

    # 🔥 构建 schedule
    steering_schedule = None
    if steering_embeddings is not None:
        steering_schedule = build_steering_schedule(unet, steering_embeddings)
        print("Steering timesteps:", list(steering_schedule.keys()))

    for t in tqdm(scheduler.timesteps):

        handles = []

        t_value = int(t.item())

        # 🔥 只在 load 到的 timestep 才 steer
        if steering_schedule is not None:
            handles = register_steering_hooks_for_timestep(
                unet,
                steering_schedule,
                t_value,
                steering_strength
            )

        latent_model_input = torch.cat([latents] * 2)
        text_emb = torch.cat([unconditional_embeddings, text_embeddings])

        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_emb
            ).sample

        noise_pred_uncon, noise_pred_con = noise_pred.chunk(2)
        noise_pred = noise_pred_uncon + guidance_scale * (
            noise_pred_con - noise_pred_uncon
        )

        latents, pred_x0 = step(scheduler, noise_pred, t, latents)

        # 🔥 移除 hooks
        for h in handles:
            h.remove()

    return latents