import argparse
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
from transformers import AutoTokenizer, AutoModel


def _iter_attention_modules(unet):
    """Yield (idx, name, module) for each attention-like submodule."""
    count = 0
    for name, module in unet.named_modules():
        lname = name.lower()
        if "attn2" in lname :
            yield count, name, module
            count += 1


def get_text_embedding(tokenizer, text_encoder, prompts, device):
    text_input = tokenizer(
        prompts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    return text_encoder(text_input.input_ids.to(device))[0]


def extract_embeddings(concepts: list,
                       layer_idxs: list,
                       timesteps: list,
                       save_path: str,
                       model_name: str = "CompVis/stable-diffusion-v1-4",
                       num_inference_steps: int = 50,
                       guidance_scale: float = 7.5,
                       embeddings_per_template: int = 1,
                       device: str = None,
                       use_fp16: bool = False):

    templates = [
        "a photo of {}, without any background",
        "a picture of {}, without any background",
        "a rendering of {}, without any background",
        "a sketch of {}, without any background",
        "an illustration of {}, without any background",
        "an image of {}, without any background",
    ]

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    print(f"loading model {model_name} on {device}")
    unet = UNet2DConditionModel.from_pretrained(
        model_name, subfolder="unet"
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, subfolder="tokenizer"
    )

    text_encoder = AutoModel.from_pretrained(
        model_name, subfolder="text_encoder"
    ).to(device)

    if use_fp16 and device.type == "cuda":
        print("converting models to fp16 to save VRAM")
        unet = unet.half()
        text_encoder = text_encoder.half()

    scheduler = DDIMScheduler.from_pretrained(
        model_name, subfolder="scheduler"
    )

    if embeddings_per_template < 1:
        raise ValueError("embeddings_per_template must be >= 1")

    latent_shape = (
        1,
        unet.config.in_channels,
        unet.config.sample_size,
        unet.config.sample_size,
    )

    scheduler.set_timesteps(num_inference_steps)

    captured = {}
    schedule_set = set(timesteps)

    # =====================================================
    # 🔥 构建 attn2 的 index -> (name, module) 映射
    # =====================================================
    attention_modules = []
    for idx, (name, module) in enumerate(
            [(n, m) for n, m in unet.named_modules()
             if "attn2" in n.lower()]):
        attention_modules.append((idx, name, module))

    # 构建 index -> name/module dict
    index_to_name = {}
    index_to_module = {}

    for idx, name, module in attention_modules:
        index_to_name[idx] = name
        index_to_module[idx] = module

    print("Detected attn2 modules:")
    for idx in index_to_name:
        print(f"  [{idx}] {index_to_name[idx]}")

    print("beginning denoising loop")

    with torch.no_grad():

        for concept in concepts:
            print(f"processing concept: {concept}")
            captured[concept] = {}

            for template in templates:

                prompt_text = template.format(concept)
                print(f"  using template: '{prompt_text}'")

                prompt_emb = get_text_embedding(
                    tokenizer,
                    text_encoder,
                    [prompt_text],
                    device
                )

                if use_fp16 and device.type == "cuda":
                    prompt_emb = prompt_emb.half()

                captured[concept][template] = {}

                for repeat_idx in range(embeddings_per_template):
                    lat_dtype = torch.float16 if (use_fp16 and device.type == "cuda") else torch.float32
                    lat = torch.randn(latent_shape, device=device, dtype=lat_dtype)
                    captured[concept][template][repeat_idx] = {}

                    for i, t in enumerate(scheduler.timesteps):

                        ti = i
                        tval = int(t.item() if hasattr(t, "item") else t)

                        handles = []

                        if ti in schedule_set:

                            print(
                                f"    capture repeat {repeat_idx} at index {ti} (timestep {tval})"
                            )

                            for idx in layer_idxs:

                                if idx not in index_to_module:
                                    raise ValueError(
                                        f"attn2 index {idx} not found"
                                    )

                                module_name = index_to_name[idx]
                                module = index_to_module[idx]

                                def make_hook(module_full_name,
                                              ts_index,
                                              ts_val,
                                              rep,
                                              conc=concept,
                                              templ=template):

                                    def hook(module, inputs, output):

                                        captured \
                                            .setdefault(conc, {}) \
                                            .setdefault(templ, {}) \
                                            .setdefault(rep, {}) \
                                            .setdefault(ts_index, {}) \
                                            [module_full_name] = {
                                                "value": ts_val,
                                                "tensor": output.detach().cpu()
                                            }

                                    return hook

                                handles.append(
                                    module.register_forward_hook(
                                        make_hook(module_name, ti, tval, repeat_idx)
                                    )
                                )

                        # UNet forward
                        noise_pred = unet(
                            lat,
                            t,
                            encoder_hidden_states=prompt_emb
                        ).sample

                        # DDIM step
                        prev_timestep = (
                            t - scheduler.config.num_train_timesteps
                            // scheduler.num_inference_steps
                        )

                        alpha_prod_t = scheduler.alphas_cumprod[t]
                        alpha_prod_t_prev = (
                            scheduler.alphas_cumprod[prev_timestep]
                            if prev_timestep > 0
                            else scheduler.final_alpha_cumprod
                        )

                        beta_prod_t = 1 - alpha_prod_t

                        pred_x0 = (
                            lat - beta_prod_t**0.5 * noise_pred
                        ) / alpha_prod_t**0.5

                        pred_dir = (
                            (1 - alpha_prod_t_prev) ** 0.5
                        ) * noise_pred

                        lat = (
                            alpha_prod_t_prev**0.5 * pred_x0
                            + pred_dir
                        )

                        for h in handles:
                            h.remove()

                if device.type == "cuda":
                    torch.cuda.empty_cache()

    print(f"writing captured embeddings to {save_path}")
    torch.save(captured, save_path)
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract attention embeddings during diffusion.")
    parser.add_argument("--concept", type=str, nargs='+', required=True,
                        help="One or more prompt strings (concepts) to encode")
    parser.add_argument("--layers", type=int, nargs="+", required=False, default=[4, 5, 6, 11, 12, 13, 18, 19, 20, 25, 26, 27, 32, 33, 34, 39, 40, 41, 46, 47, 48, 53, 54, 55, 60, 61, 62, 67, 68, 69, 74, 75, 76, 81, 82, 83, 88, 89, 90, 95, 96, 97, 102, 103, 104, 109, 110, 111])
    parser.add_argument("--timesteps", type=int, nargs="+", required=True)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--model", type=str, default="/root/autodl-tmp/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--embeddings-per-template", type=int, default=1,
                        help="How many embeddings to extract per prompt template")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (cpu or cuda).  Default auto-selects.")
    parser.add_argument("--fp16", action="store_true",
                        help="Convert model and latents to float16 to save VRAM")
    args = parser.parse_args()

    extract_embeddings(
        args.concept,
        args.layers,
        args.timesteps,
        args.save,
        model_name=args.model,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        embeddings_per_template=args.embeddings_per_template,
        device=args.device,
        use_fp16=args.fp16,
    )
