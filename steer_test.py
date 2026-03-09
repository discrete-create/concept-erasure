import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
from transformers import AutoTokenizer, AutoModel

from gen_imag import (
    gen_image,
    get_text_embedding,
    latent2image,
)

def get_attn2_name_mapping(unet):
    """
    返回:
        idx2name: {0: full_name, 1: full_name, ...}
    """
    idx2name = {}
    layer_idx = 0
    attn2_out_list = []
    for name, module in unet.named_modules():
        if "attn2" in name.lower():
            idx2name[layer_idx] = name
            print(f"[{layer_idx}] {name}")
            layer_idx += 1
            if "out" in name.lower():
                attn2_out_list.append(layer_idx-1)

    print(f"\nTotal attn2 layers: {layer_idx}")
    print(f"attn2 out layers: {attn2_out_list}")
    return idx2name


def main():
    # configuration
    model_name = "/root/autodl-tmp/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # load models
    print(f"loading model {model_name} on {device}")
    unet = UNet2DConditionModel.from_pretrained(
        model_name, subfolder="unet"
    ).to(device)

    # 打印并获取 attn2 mapping
    idx2name = get_attn2_name_mapping(unet)

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = AutoModel.from_pretrained(
        model_name, subfolder="text_encoder"
    ).to(device)
    scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

    # reduce memory
    if device.type == "cuda":
        print("converting models to fp16")
        unet = unet.half()
        vae = vae.half()
        text_encoder = text_encoder.half()

    # prompt
    concept = "cat"
    prompt = "a photo of a cat"
    prompt_emb = get_text_embedding(tokenizer, text_encoder, [prompt], device)

    # load steering bank (按 full_name 存的)
    embs = torch.load("embs.pt")

    # random latents
    latents = torch.randn(
        (1, unet.config.in_channels, 64, 64),
        device=device,
    )
    if device.type == "cuda":
        latents = latents.half()

    scheduler.set_timesteps(50)

    print("generating baseline image")
    base_lat = gen_image(
        latents.clone(),
        prompt_emb,
        tokenizer,
        text_encoder,
        unet,
        scheduler,
        device,
        guidance_scale=7.5,
    )

    base_img = latent2image(vae, base_lat)
    from PIL import Image
    Image.fromarray(base_img).save("baseline.png")
    print("baseline image saved")

    torch.cuda.empty_cache()

    # ========= 核心修改部分 =========

    # 你传的是编号
    selected_indices = [0, 5, 10, 15, 20]

    # 转换为 full_name
    selected_names = [idx2name[i] for i in selected_indices]

    print("\nSelected layers for steering:")
    for i, name in zip(selected_indices, selected_names):
        print(f"{i} -> {name}")

    # =================================

    steered_lat = gen_image(
        latents.clone(),
        prompt_emb,
        tokenizer,
        text_encoder,
        unet,
        scheduler,
        device,
        guidance_scale=7.5,
        steering_embeddings=embs,
        steering_strength=15
    )

    steered_img = latent2image(vae, steered_lat)
    Image.fromarray(steered_img).save("erased.png")
    print("steered image saved")
    print("hooks removed")

    print(
        f"baseline image shape {base_img.shape}, "
        f"erasure image shape {steered_img.shape}"
    )


if __name__ == "__main__":
    main()