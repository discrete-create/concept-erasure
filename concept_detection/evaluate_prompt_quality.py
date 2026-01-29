import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any
from transformers import CLIPTokenizer, CLIPTextModel
#from build_unsafe_matrix import encode_prompts
'''
def encode_prompts(prompts, tokenier,text_encoder,batch_size=256):
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
'''
def evaluate_unsafe_prototypes_with_feedback(
    prompt_emb_pairs: List[Tuple[str, torch.Tensor]],
    sim_high: float = 0.9,
    sim_mid: float = 0.75,
    min_effective_prototypes: int = 3,
):
    """
    Evaluate unsafe prototype quality and produce agent-usable feedback.

    Args:
        prompt_emb_pairs: List[(prompt, embedding)]
        sim_high: similarity threshold for collapse
        sim_mid: similarity threshold for coverage
        min_effective_prototypes: minimum usable modes

    Returns:
        metrics: numeric diagnostics
        feedback: structured feedback for agent
        passed: bool
    """

    prompts = [p for p, _ in prompt_emb_pairs]
    U = torch.stack([e for _, e in prompt_emb_pairs]).float()
    U = F.normalize(U, dim=-1)
    N = U.shape[0]

    metrics: Dict[str, Any] = {}
    feedback: Dict[str, Any] = {
        "status": "pass",
        "failure_modes": [],
        "prompt_tags": {},
        "recommendations": [],
        "examples": {}
    }

    if N < min_effective_prototypes:
        feedback["status"] = "fail"
        feedback["failure_modes"].append("too_few_prototypes")
        feedback["recommendations"].append(
            "generate more semantically distinct unsafe prompts"
        )
        return metrics, feedback, False

    # ---------- similarity matrix ----------
    sim_mat = torch.matmul(U, U.T)
    eye_mask = torch.eye(N, device=U.device).bool()

    # ---------- 1️⃣ collapse / over-similarity ----------
    pairwise = sim_mat[~eye_mask]
    metrics["pairwise_sim_max"] = pairwise.max().item()

    if metrics["pairwise_sim_max"] >= sim_high:
        feedback["status"] = "fail"
        feedback["failure_modes"].append("prototype_collapse")

        # tag prompts involved in high similarity
        high_sim_pairs = (sim_mat >= sim_high) & (~eye_mask)
        for i in range(N):
            if high_sim_pairs[i].any():
                feedback["prompt_tags"].setdefault(i, []).append("too_similar")

        feedback["recommendations"].append(
            "generate prompts with different semantic framing, intent, or abstraction level"
        )

    # ---------- 2️⃣ effective prototype count ----------
    effective = (sim_mat < sim_high).any(dim=1)
    metrics["effective_prototypes"] = int(effective.sum().item())

    if metrics["effective_prototypes"] < min_effective_prototypes:
        feedback["status"] = "fail"
        feedback["failure_modes"].append("insufficient_modes")

        under = (~effective).nonzero(as_tuple=True)[0].tolist()
        feedback["examples"]["over_represented"] = [prompts[i] for i in under[:3]]

        feedback["recommendations"].append(
            "introduce new unsafe prompts that differ in scenario, subject, or intent"
        )

    # ---------- 3️⃣ dominance analysis ----------
    sim_no_self = sim_mat - torch.eye(N, device=U.device) * 10.0
    nearest = torch.argmax(sim_no_self, dim=1)
    counts = torch.bincount(nearest, minlength=N)
    dominant_idx = torch.argmax(counts).item()
    dominant_ratio = counts[dominant_idx].item() / N

    metrics["dominant_proto_ratio"] = dominant_ratio

    if dominant_ratio >= 0.5:
        feedback["status"] = "fail"
        feedback["failure_modes"].append("single_mode_dominance")

        dominated = (nearest == dominant_idx).nonzero(as_tuple=True)[0].tolist()
        feedback["examples"]["over_represented"] = [prompts[i] for i in dominated[:3]]

        feedback["recommendations"].append(
            f"avoid generating prompts similar to: '{prompts[dominant_idx]}'"
        )

    # ---------- 4️⃣ max-sim spread (gradient usability) ----------
    max_sim = sim_no_self.max(dim=1).values
    spread = torch.quantile(max_sim, 0.9) - torch.quantile(max_sim, 0.1)
    metrics["max_sim_spread"] = spread.item()

    if spread < 0.15:
        feedback["status"] = "fail"
        feedback["failure_modes"].append("insufficient_gradient_range")

        feedback["recommendations"].append(
            "generate prompts with varying degrees of explicitness and abstraction"
        )

    passed = feedback["status"] == "pass"
    metrics["passed"] = passed

    return metrics, feedback, passed

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any

def spatial_contribution_prune(
    old_pairs: List[Tuple[str, torch.Tensor]],
    new_pairs: List[Tuple[str, torch.Tensor]],
    sim_high: float = 0.9,
    sim_mid: float = 0.75,
    max_keep: int = 500,
):
    """
    Merge old + new unsafe prompts, keeping only those that
    contribute to embedding space coverage.

    Returns:
        kept_pairs: List[(prompt, embedding)]
        prune_feedback: dict for agent
    """
    prune_feedback: Dict[str, Any] = {
        "kept_new": [],
        "dropped_new": [],
        "reasons": {}
    }

    # ---------- prepare ----------
    kept_pairs = list(old_pairs)

    if len(old_pairs) > 0:
        U_old = torch.stack([e for _, e in old_pairs]).float()
        U_old = F.normalize(U_old, dim=-1)
        sim_old = torch.matmul(U_old, U_old.T)
        dist_old = 1 - sim_old
        current_diameter = dist_old.max().item()
    else:
        # 空集合初始化
        U_old = None
        current_diameter = 0.0

    # ---------- evaluate each new prompt ----------
    for prompt, emb in new_pairs:
        e = F.normalize(emb.float(), dim=-1)

        if U_old is None:
            keep = True
            reasons = ["initial_prompt"]
        else:
            sims = torch.matmul(U_old, e)
            max_sim = sims.max().item()
            min_sim = sims.min().item()

            novelty = 1 - max_sim
            diam_gain = (1 - min_sim) - current_diameter
            new_mode = max_sim < sim_mid

            keep = False
            reasons = []

            if novelty > (1 - sim_high):
                keep = True
                reasons.append("novel_semantic_region")

            if diam_gain > 0.05:
                keep = True
                reasons.append("expanded_embedding_diameter")

            if new_mode:
                keep = True
                reasons.append("new_semantic_mode")

        if keep:
            kept_pairs.append((prompt, emb))
            prune_feedback["kept_new"].append(prompt)
            prune_feedback["reasons"][prompt] = reasons

            # update U_old incrementally
            if U_old is None:
                U_old = e.unsqueeze(0)
            else:
                U_old = torch.cat([U_old, e.unsqueeze(0)], dim=0)
                current_diameter = max(
                    current_diameter,
                    (1 - (torch.matmul(U_old[:-1], e).min())).item()
                )
        else:
            prune_feedback["dropped_new"].append(prompt)
            prune_feedback["reasons"][prompt] = ["low_spatial_contribution"]

    # ---------- size control ----------
    if len(kept_pairs) > max_keep:
        kept_pairs = kept_pairs[-max_keep:]

    return kept_pairs, prune_feedback

from typing import List, Tuple
import torch

def build_prompt_embedding_pairs(
    prompts: List[str],
    encode_prompts,
    tokenier,
    text_encoder
) -> List[Tuple[str, torch.Tensor]]:
    """
    Build (prompt, embedding) pairs from prompt list.

    Args:
        prompts: list of prompt strings
        encode_prompts: function(prompts) -> Tensor [N, D]

    Returns:
        List[(prompt, embedding)]
    """
    with torch.inference_mode():
        embeddings = encode_prompts(prompts,tokenier,text_encoder)

    assert isinstance(embeddings, torch.Tensor), \
        "encode_prompts must return a torch.Tensor"
    assert embeddings.shape[0] == len(prompts), \
        "number of embeddings must match number of prompts"

    pairs = [
        (prompt, embeddings[i])
        for i, prompt in enumerate(prompts)
    ]

    return pairs
