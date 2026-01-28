import os
import json
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel

# ----------------------------
# 配置
# ----------------------------
INPUT_JSON = "prompt.json"
SAVE_DIR = "unsafe_embeddings"
CLIP_MODEL_Tokennider = "/root/autodl-tmp/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b/tokenizer"
CLIP_MODEL_Text_encoder="/root/autodl-tmp/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b/text_encoder"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------
# Load CLIP text encoder
# ----------------------------
tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_Tokennider)
text_encoder = CLIPTextModel.from_pretrained(CLIP_MODEL_Text_encoder).to(DEVICE)
text_encoder.eval()

@torch.no_grad()
def encode_prompts(prompts, batch_size=256):
    embeddings = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i:i + batch_size]

        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(DEVICE)

        outputs = text_encoder(**tokens)
        # CLIP 使用 last_hidden_state 的 [EOS] token
        emb = outputs.last_hidden_state[:, -1, :]  # [B, D]
        emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize

        embeddings.append(emb.cpu())

    return torch.cat(embeddings, dim=0)
# ----------------------------
# Load data
# ----------------------------
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# label -> list[prompt]
label_to_prompts = {}

for item in data:
    prompt = item["prompt"]
    labels = item["label"]

    for lbl in labels:
        label_to_prompts.setdefault(lbl, []).append(prompt)

# ----------------------------
# Build & save matrices
# ----------------------------
for label, prompts in label_to_prompts.items():
    print(f"Processing label: {label} ({len(prompts)} prompts)")

    emb = encode_prompts(prompts)

    save_path = os.path.join(SAVE_DIR, f"{label}.pt")

    torch.save(
        {
            "label": label,
            "embeddings": emb,        # [N, D]
            "num_prompts": len(prompts),
            "dim": emb.shape[1]
        },
        save_path
    )

    print(f"Saved -> {save_path}")
