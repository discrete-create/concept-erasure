import argparse
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
from transformers import AutoTokenizer, AutoModel


def _iter_attention_modules(unet):
    """Yield (idx, name, module) for each attention-like submodule."""
    count = 0
    for name, module in unet.named_modules():
        lname = name.lower()
        if "attn" in lname or "attention" in lname:
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
                       device: str = None,
                       use_fp16: bool = False):
    # fixed set of prompt templates; concept will be formatted into each
    templates = [
        "a photo of {}, without any background",
        "a picture of {}, without any background",
        "a rendering of {}, without any background",
        "a sketch of {}, without any background",
        "an illustration of {}, without any background",
        "an image of {}, without any background",
    ]
    """Run diffusion generation(s) and capture attention outputs for prompts.

    The function loops over the provided ``concepts`` (a list of strings).  For
    each one it performs a denoising pass and records the hidden state after
    the specified cross-attention layers at the requested timesteps.  Results
    are stored in a nested dictionary keyed first by concept, then timestep,
    then layer index, e.g.::

        captured[concept][timestep][layer_idx] = tensor

    The final dict is written to ``save_path`` via ``torch.save``.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    print(f"loading model {model_name} on {device}")
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = AutoModel.from_pretrained(model_name, subfolder="text_encoder").to(device)

    if use_fp16 and device.type == "cuda":
        print("converting models to fp16 to save VRAM")
        unet = unet.half()
        text_encoder = text_encoder.half()

    # we will compute embeddings separately per concept inside the loop below

    scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
    latents = torch.randn(
        (1, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size),
        device=device,
    )

    scheduler.set_timesteps(num_inference_steps)

    # storage: {concept: {timestep: {layer_idx: tensor}}}
    captured = {}
    # treat provided timesteps list as indices into scheduler.timesteps
    schedule_set = set(timesteps)

    # helper to look up module reference by layer index
    attention_modules = list(_iter_attention_modules(unet))
    def _get_module(idx):
        for i, name, mod in attention_modules:
            if i == idx:
                return mod
        raise IndexError(f"no attention module at index {idx}")

    print("beginning denoising loop for each concept and template")
    with torch.no_grad():                          # disable autograd to reduce memory
        for concept in concepts:
            print(f"processing concept: {concept}")
            captured[concept] = {}
            for template in templates:
                prompt_text = template.format(concept)
                print(f"  using template: '{prompt_text}'")
                # compute text embedding for this prompt
                prompt_emb = get_text_embedding(tokenizer, text_encoder, [prompt_text], device)
                if use_fp16 and device.type == "cuda":
                    prompt_emb = prompt_emb.half()
                # reset latents for each prompt/template
                lat = latents.clone().to(device)
                if use_fp16 and device.type == "cuda":
                    lat = lat.half()

                # store a slot for this template
                captured[concept][template] = {}

                for i, t in enumerate(scheduler.timesteps):
                    # index and actual timestep value
                    ti = i
                    tval = int(t.item() if hasattr(t, "item") else t)
                    handles = []
                    if ti in schedule_set:
                        print(f"    installing capture hooks at index {ti} (timestep {tval})")
                        for idx in layer_idxs:
                            mod = _get_module(idx)

                            def make_hook(layer_idx, ts_index, ts_val, conc=concept, templ=template):
                                def hook(module, inputs, output):
                                    # store using both index and value for clarity
                                    captured.setdefault(conc, {}).setdefault(templ, {}).setdefault(ts_index, {})[layer_idx] = {
                                        'value': ts_val,
                                        'tensor': output.detach().cpu()
                                    }
                                return hook

                            handles.append(mod.register_forward_hook(make_hook(idx, ti, tval)))
                    # normal UNet forward
                    noise_pred = unet(lat, t, encoder_hidden_states=prompt_emb).sample
                    # (we ignore guidance for simplicity; user can modify as needed)

                    # simple DDIM step calculation
                    prev_timestep = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
                    alpha_prod_t = scheduler.alphas_cumprod[t]
                    alpha_prod_t_prev = (
                        scheduler.alphas_cumprod[prev_timestep]
                        if prev_timestep > 0
                        else scheduler.final_alpha_cumprod
                    )
                    beta_prod_t = 1 - alpha_prod_t
                    pred_x0 = (lat - beta_prod_t**0.5 * noise_pred) / alpha_prod_t**0.5
                    pred_dir = (1 - alpha_prod_t_prev) ** 0.5 * noise_pred
                    lat = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir

                    # remove hooks
                    for h in handles:
                        h.remove()

                # free any cached GPU memory after each template
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    print(f"writing captured embeddings to {save_path}")
    torch.save(captured, save_path)
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract attention embeddings during diffusion.")
    parser.add_argument("--concept", type=str, nargs='+', required=True,
                        help="One or more prompt strings (concepts) to encode")
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--timesteps", type=int, nargs="+", required=True)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--model", type=str, default="/root/autodl-tmp/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=7.5)
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
        device=args.device,
        use_fp16=args.fp16,
    )
