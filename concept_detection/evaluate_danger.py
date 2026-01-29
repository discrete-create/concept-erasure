import os
import torch
def load_label_matrix(label, base_dir="unsafe_embeddings"):
    path = os.path.join(base_dir, f"{label}.pt")
    data = torch.load(path, map_location="cpu")
    return data["embeddings"]  # [N, D]

@torch.no_grad()
def encode_single_prompt(prompt: str,tokenizer, text_encoder, DEVICE='cuda'):
    tokens = tokenizer(
        [prompt],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt"
    ).to(DEVICE)

    out = text_encoder(**tokens)
    emb = out.last_hidden_state[:, -1, :]
    emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb.cpu()  # [1, D]

@torch.no_grad()
def encode_batch_prompts(prompts,tokenizer, text_encoder, DEVICE='cuda'):
    tokens = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt"
    ).to(DEVICE)

    out = text_encoder(**tokens)
    emb = out.last_hidden_state[:, -1, :]
    emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb.cpu()  # [B, D]

import torch
import numpy as np
def extract_danger_labels(sim_dict, threshold=1.0):
    """
    sim_dict: {label_name: z_score}
    return: {label_name: z_score} (z_score >= threshold)
    """
    return {
    label: z
    for label, z in sim_dict.items()
    if z >= threshold
    }
def build_danger_subspace(unsafe_matrix, k=16):
    """
    unsafe_matrix: [N, D] tensor
    k: PCA 保留主成分数量
    return: D [k, D] 主方向矩阵
    """
    # 转为 numpy
    X = unsafe_matrix.numpy()
    
    # 中心化
    X_mean = X.mean(axis=0, keepdims=True)
    X_centered = X - X_mean

    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # 前 k 个主方向
    D = Vt[:k, :]  # shape [k, D]
    return torch.tensor(D, dtype=torch.float32), torch.tensor(X_mean, dtype=torch.float32)
def danger_projection_score(prompt_emb, D, mean_vec):
    """
    prompt_emb: [1, D]
    D: [k, D] 危险主方向矩阵
    mean_vec: [1, D] unsafe 矩阵均值
    return: projection norm (越大越危险)
    """
    x_centered = prompt_emb - mean_vec
    proj = torch.matmul(x_centered, D.T)  # [1, k]
    score = torch.norm(proj, p=2).item()
    return score

import torch
import torch.nn.functional as F

def compute_centroid(unsafe_matrix: torch.Tensor):
    """
    unsafe_matrix: [N, D]  (未 normalize 或 normalize 都可以)
    return:
        centroid: [D] (L2 normalized)
        mu: [D]       (未 normalize 的均值，用于 PCA / center)
    """

    # 1. 先算“真实均值”（不要提前 normalize）
    mu = unsafe_matrix.mean(dim=0)  # [D]

    # 2. 再归一化成方向向量（CLIP 语义方向）
    centroid = F.normalize(mu, dim=-1)

    return centroid, mu
def remove_danger_component(prompt_emb, D, mean_vec, tau=0.2, normalize=True):
    """
    将 prompt embedding 在危险 PCA 子空间上的分量进行【软删除】
    
    Args:
        prompt_emb: (d,) 原始 prompt embedding
        D: (k, d) PCA 得到的危险子空间正交基（行向量，单位正交）
        mean_vec: (d,) 用于 PCA 的均值向量
        tau: 软删除阈值，控制删除强度
        normalize: 是否对输出 embedding 做 L2 normalize

    Returns:
        x_safe: (d,) 安全化后的 embedding
    """

    # 1. 中心化
    x_centered = prompt_emb - mean_vec

    # 2. 危险子空间投影（PCA 基是正交的）
    proj = torch.matmul(
        torch.matmul(x_centered, D.T),  # (k,)
        D                              # (d,)
    )  # (d,)

    # 3. 软删除系数（根据危险强度）
    proj_norm = torch.norm(proj)
    alpha = torch.clamp(proj_norm / tau, min=0.0, max=1.0)

    # 4. 软删除危险分量
    x_safe = x_centered - alpha * proj

    # 5. 加回均值
    x_safe = x_safe + mean_vec

    # 6. 可选 normalize（强烈推荐）
    if normalize:
        x_safe = F.normalize(x_safe, dim=-1)

    return x_safe
def max_similarity(prompt_emb, unsafe_matrix):
    """
    prompt_emb: [1, D]
    unsafe_matrix: [N, D]
    """
    sims = torch.matmul(prompt_emb, unsafe_matrix.T)  # [1, N]
    return sims.max().item()


import os
import torch
import torch.nn.functional as F


def prompt_to_label_similarity(prompt_emb, label_dirs="unsafe_embeddings"):
    """
    输入一个 prompt embedding
    计算它相对于每个 unsafe concept 的 z-score 相似度
    返回：{ label_name : z_score }
    """

    similarity_dict = {}

    # 保证 prompt_emb 是 [1, D]
    if prompt_emb.dim() == 1:
        prompt_emb = prompt_emb.unsqueeze(0)

    prompt_emb = F.normalize(prompt_emb, dim=-1)

    for file_name in os.listdir(label_dirs):
        if not file_name.endswith(".pt"):
            continue

        label_name = file_name.replace(".pt", "")
        unsafe_data = torch.load(
            os.path.join(label_dirs, file_name),
            map_location="cpu"
        )
        unsafe_matrix = unsafe_data["embeddings"]   # [N, D]

        # ensure embeddings use float32 (avoid double vs float mismatches)
        unsafe_matrix = unsafe_matrix.float()

        # compute centroid from the (float) matrix
        centroid, _ = compute_centroid(unsafe_matrix)
        centroid = F.normalize(centroid, dim=-1)
        unsafe_matrix = F.normalize(unsafe_matrix, dim=-1)

        # ===== 1. sim(p, centroid) =====
        sim_p = torch.max(
            torch.matmul(prompt_emb, unsafe_matrix.T)
        ).item()
        # ===== 2. μ_C, σ_C =====
        sims = torch.matmul(unsafe_matrix, centroid)  # [N]
        mu_sim = sims.mean().item()
        sigma = sims.std(unbiased=False).item() + 1e-6

        # ===== 3. z-score =====
        z_score = (sim_p - mu_sim) / sigma

        similarity_dict[label_name] = z_score

    return similarity_dict

def mean_text_embedding(safe_prompt_list, normalize=True):
    """
    safe_prompt_list: List[Tensor]
        each tensor is [77, 768] or [1, 77, 768]

    return:
        mean_embedding: [77, 768]
    """

    # 1. 统一 shape -> [77, 768]
    embeds = []
    for emb in safe_prompt_list:
        if emb.dim() == 3:
            emb = emb.squeeze(0)   # [1, 77, 768] -> [77, 768]
        embeds.append(emb)

    # 2. stack -> [N, 77, 768]
    stacked = torch.stack(embeds, dim=0)

    # 3. 对 prompt 维度求均值
    mean_emb = stacked.mean(dim=0)  # [77, 768]

    # 4. 可选：token-level normalize（推荐）
    if normalize:
        mean_emb = F.normalize(mean_emb, dim=-1)

    return mean_emb
@torch.no_grad()
def sanitize_token_embeddings(
    full_hidden,      # [1, 77, 768]
    eos_safe,         # [1, 768]  安全化后的 EOS
    eos_orig,         # [1, 768]  原始 EOS
    strength=0.3
):
    """
    用 EOS 的“修正方向”对整句 token embedding 做小幅修正
    """
    # 方向差
    delta = eos_safe - eos_orig  # [1, 768]

    # 只对非 padding token 施加
    delta = delta.unsqueeze(1)   # [1, 1, 768]

    # 小步注入（不破分布）
    hidden_safe = full_hidden + strength * delta

    return hidden_safe