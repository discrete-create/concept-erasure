import torch
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
from get_unsafe_prompt import generate_unsafe_prompts
from evaluate_prompt_quality import *
from evaluate_danger import max_similarity
from transformers import CLIPTokenizer, CLIPTextModel
from typing import List, Tuple, Dict, Any
CLIP_MODEL_Tokennider = "/root/autodl-tmp/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b/tokenizer"
CLIP_MODEL_Text_encoder="/root/autodl-tmp/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b/text_encoder"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Load CLIP text encoder
# ----------------------------
tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_Tokennider)
text_encoder = CLIPTextModel.from_pretrained(CLIP_MODEL_Text_encoder).to(DEVICE)
text_encoder.eval()
def encode_prompts(prompts,tokenizer,text_encoder,batch_size=256):
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

def autonomous_iterative_prompt_generation(
    concept: str,
    target_num: int,
    encode_prompts,
    max_global_rounds: int = 100,
    per_round_generate: int = 40,
    sim_high: float = 0.9,
    sim_mid: float = 0.75,
):
    """
    Fully autonomous unsafe prompt generator with self-feedback loop.

    Returns:
        final_prompts: List[str]
        diagnostics: Dict
    """

    kept_pairs: List[Tuple[str, torch.Tensor]] = []
    diagnostics: Dict[str, Any] = {
        "rounds": [],
        "final_passed": False,
    }

    feedback_memory = []  # textual feedback to steer future generations

    for global_round in range(max_global_rounds):
        print(f"\n=== Global Round {global_round + 1} ===")

        # ---------- 1️⃣ generate ----------
        steer_text = ""
        if feedback_memory:
            steer_text = (
                "\n\n请特别注意以下改进建议（用于避免语义坍缩）：\n"
                + "\n".join(f"- {x}" for x in feedback_memory[-3:])
            )

        raw_prompts = generate_unsafe_prompts(
            concept + steer_text,
            target_num=per_round_generate,
        )

        if not raw_prompts:
            continue

        # ---------- 2️⃣ encode ----------
        new_pairs = build_prompt_embedding_pairs(
            raw_prompts,
            encode_prompts,
            tokenizer,text_encoder
        )
        
        # ---------- 3️⃣ prune by spatial contribution ----------
        with torch.no_grad():
            kept_pairs, prune_feedback = spatial_contribution_prune(
                old_pairs=kept_pairs,
                new_pairs=new_pairs,
                sim_high=sim_high,
                sim_mid=sim_mid,
                max_keep=target_num * 2,
            )

        # ---------- 4️⃣ evaluate ----------
        with torch.no_grad():
            metrics, feedback, passed = evaluate_unsafe_prototypes_with_feedback(
                kept_pairs,
                sim_high=sim_high,
                sim_mid=sim_mid,
            )

        diagnostics["rounds"].append({
            "metrics": metrics,
            "feedback": feedback,
            "kept": len(kept_pairs),
        })

        # ---------- 5️⃣ update feedback memory ----------
        if not passed:
            feedback_memory=feedback.get("recommendations", [])
        else:
            diagnostics["final_passed"] = True

        # ---------- 6️⃣ early stop if enough ----------
        if len(kept_pairs) >= target_num and passed:
            break
        del new_pairs
        del raw_prompts

    final_prompts = [p for p, _ in kept_pairs][:target_num]
    return final_prompts, diagnostics

Concept_list = ["blood", "Drug crime", "murder", "terrifying content"]

os.makedirs("./generated", exist_ok=True)

for concept in Concept_list:
    print(f"\n=== Generating concept: {concept} ===")

    prompts, diagnose = autonomous_iterative_prompt_generation(
        concept,
        1000,
        encode_prompts
    )

    # 文件名安全化（空格 / 特殊字符）
    print(diagnose)
    safe_name = concept.lower().replace(" ", "_")
    filename = f"./generated/{safe_name}_1000.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            {
                "concept": concept,
                "prompts": prompts,
                "diagnostics": diagnose
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Saved {len(prompts)} items to {filename}")