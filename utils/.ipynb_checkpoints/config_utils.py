import copy
import torch
import numpy as np
from tqdm import tqdm
import gc

import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import UNet2DConditionModel
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

def import_models_for_cfr(pretrained_model_name_or_path,device,dtype=torch.float32):
    tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
    use_fast=False,
    )
    text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name_or_path)
    text_encoder = text_encoder_cls.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet"
    )
    text_encoder.to(device=device, dtype=dtype)
    unet.to(device,dtype=dtype)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    return tokenizer,text_encoder,unet

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")