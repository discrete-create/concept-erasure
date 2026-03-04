import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
from transformers import AutoTokenizer, AutoModel

from gen_imag import (
    gen_image,
    get_text_embedding,
    latent2image,
    register_steering_hooks,
)


def main():
    # configuration
    model_name = "/root/autodl-tmp/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # load models
    print(f"loading model {model_name} on {device}")
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(device)
    # load associated VAE for decoding
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = AutoModel.from_pretrained(model_name, subfolder="text_encoder").to(device)
    scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

    # reduce memory by switching to float16 on GPU
    if device.type == "cuda":
        print("converting models to fp16 for memory savings")
        unet = unet.half()
        vae = vae.half()
        text_encoder = text_encoder.half()

    # pick a prompt for concept erasure test
    concept = "cat"
    prompt = f"a photo of {concept}, without any background"
    prompt_emb = get_text_embedding(tokenizer, text_encoder, [prompt], device)
    # choose one of the templates stored in embs.pt
    embs = torch.load("embs.pt")

    # random starting latents
    latents = torch.randn(
        (1, unet.config.in_channels, 64, 64),
        device=device,
    )
    if device.type == "cuda":
        latents = latents.half()

    # run baseline generation
    # use fewer steps to reduce memory and runtime
    scheduler.set_timesteps(50)
    print("generating baseline image")
    base_lat = gen_image(latents.clone(), prompt_emb, tokenizer, text_encoder, unet, scheduler, device, guidance_scale=7.5)
    base_img = latent2image(vae, base_lat)
    # save array as PNG
    from PIL import Image
    Image.fromarray(base_img).save("baseline.png")
    print("baseline image saved to baseline.png")

    
    torch.cuda.empty_cache()

    # install steering hooks in opposite direction (now on correct device)
    handles = register_steering_hooks(unet, [2,5],embs, steering_strength=3)
    print(f"registered {len(handles)} steering hooks (negative direction)")

    steered_lat = gen_image(latents.clone(), prompt_emb, tokenizer, text_encoder, unet, scheduler, device, guidance_scale=7.5)
    steered_img = latent2image(vae, steered_lat)
    from PIL import Image
    Image.fromarray(steered_img).save("erased.png")
    print("steered image saved to erased.png")

    # cleanup hooks
    for h in handles:
        h.remove()
    print("hooks removed")

    # display some diagnostics
    print(f"baseline image shape {base_img.shape}, erasure image shape {steered_img.shape}")


if __name__ == "__main__":
    main()
