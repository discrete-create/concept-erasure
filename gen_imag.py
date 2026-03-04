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


def _create_steering_hook(idx, steer_tensor, strength):
    """Return a forward hook that adds a scaled steer_tensor to its output.

    ``idx`` is used for logging only; hooks themselves operate on the
    activation tensor produced by the module.
    """
    def hook(module, inputs, output):
        # attempt in-place addition for efficiency
        try:
            return output + strength * steer_tensor
        except Exception:
            return output + strength * steer_tensor
    return hook


def register_steering_hooks(unet, layers, steering_embeddings, steering_strength=1.0):
    """Install forward hooks on specific cross-attention layers of ``unet``.

    This function walks ``unet.named_modules()`` and counts modules whose
    name contains ``"attn"`` or ``"attention"`` (case-insensitive).  When the
    count matches a value in ``layers`` the corresponding steering embedding is
    bound to that module via a forward hook.  Hooks are returned so that they
    can be removed after inference.

    Parameters
    ----------
    unet : torch.nn.Module
        The UNet model used during diffusion.
    layers : list of int
        Indices of attention layers (in visitation order) to steer.
    steering_embeddings : list of torch.Tensor
        Embeddings to add after the chosen attention layers.  Must align with
        ``layers`` in length.
    steering_strength : float
        Scaling factor applied to each embedding.

    Returns
    -------
    list
        A list of ``torch.utils.hooks.RemovableHandle`` objects corresponding to
        the installed hooks.  Call ``handle.remove()`` when steering is no
        longer needed.
    """
    handles = []
    if layers is None or steering_embeddings is None:
        return handles
    layer_idx = 0
    emb_iter = iter(zip(layers, steering_embeddings))
    try:
        target_layer, target_emb = next(emb_iter)
    except StopIteration:
        return handles

    for name, module in unet.named_modules():
        lname = name.lower()
        if "attn" in lname or "attention" in lname:
            if layer_idx == target_layer:
                handle = module.register_forward_hook(
                    _create_steering_hook(layer_idx, target_emb, steering_strength)
                )
                handles.append(handle)
                try:
                    target_layer, target_emb = next(emb_iter)
                except StopIteration:
                    break
            layer_idx += 1
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
  
  
def gen_image(latents,
              prompts_emb,
              tokenizer,
              text_encoder,
              unet,
              scheduler,
              device,
              num_inference_steps=50,
              guidance_scale=7.5,
              steering_layers=None,
              steering_embeddings=None,
              steering_strength=1.0,
              steering_schedule: dict = None):
    """Generate images, optionally applying steering signals during inference.

    Steering is achieved by installing forward hooks on the specified cross-
    attention layers of ``unet``.  You can either pre-install hooks before
    the loop using :func:`register_steering_hooks`, or provide a
    ``steering_schedule`` that installs/removes hooks at particular
    timesteps.

    ``steering_schedule`` should be a mapping from timestep (int) to a tuple
    ``(layers, embeddings, strength)``.  During generation the routine will
    register the appropriate hooks immediately before calling the UNet for the
    given timestep and remove them right afterwards.  This allows steering to
    affect only a subset of time steps.

    Legacy parameters ``steering_layers`` / ``steering_embeddings`` /
    ``steering_strength`` are ignored and remain only for backwards
    compatibility.
    """
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
        unconditional_embeddings = text_encoder(unconditional_input.input_ids.to(device))[0]
      # text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)
    
    scheduler.set_timesteps(num_inference_steps)
    
    if steering_layers is not None or steering_embeddings is not None:
        print("[gen_image] warning: steering_layers/embeddings ignored; use register_steering_hooks instead")

    results = []
    # make a copy of schedule keys for fast membership testing
    schedule_keys = set(steering_schedule.keys()) if steering_schedule else set()

    for i,t in enumerate(tqdm(scheduler.timesteps)):
        handles = []
        if t in schedule_keys:  
            layers, emb, strength = steering_schedule[t]
            handles = register_steering_hooks(unet, layers, emb, strength)
        latent_model_input = torch.cat([latents] * 2)
        text_emb = torch.cat([unconditional_embeddings, text_embeddings])
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_emb).sample
        noise_pred_uncon, noise_pred_con = noise_pred.chunk(2)
        noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
        latents, pred_x0 = step(scheduler, noise_pred, t, latents)
        # remove hooks if we added them for this step
        for h in handles:
            h.remove()

        #results.append([latents,noise_pred, t])
    return latents
 