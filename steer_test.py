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
    # choose one of the templates stored in embs.pt
    embs = torch.load("embs.pt")
    tmpl = list(embs[concept].keys())[0]
    print(f"using template from file: {tmpl}")

    # select step index and corresponding embeddings
    step_index = list(embs[concept][tmpl].keys())[0]  # first step
    layer_dict = embs[concept][tmpl][step_index]
    layer_idxs = list(layer_dict.keys())
    print(f"steering layers: {layer_idxs}, step index {step_index}")

    # normally steering embeddings are added post-attention; for erasure we take negation
    steering_embeddings = [(-layer_dict[idx]["tensor"]).to(device) for idx in layer_idxs]

    # prepare prompt embedding (same prompt we will generate with)
    prompt = tmpl.format(concept)
    prompt_emb = get_text_embedding(tokenizer, text_encoder, [prompt], device)
    if device.type == "cuda":
        prompt_emb = prompt_emb.half()

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
        
    steering_embeddings = [emb.to(device) for emb in steering_embeddings]

    # install steering hooks in opposite direction (now on correct device)
    handles = register_steering_hooks(unet, layer_idxs, steering_embeddings, steering_strength=-1.5)
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
