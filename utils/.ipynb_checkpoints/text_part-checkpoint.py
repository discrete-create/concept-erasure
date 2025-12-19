import copy
import torch
import numpy as np
from tqdm import tqdm
import gc
import torch.nn as nn
from diffusers import UNet2DConditionModel
import torch.nn.functional as F
import torch.utils.checkpoint

@torch.no_grad()
def ridge_regression(W, alpha, target_features, original_features, regularization=None):
    """
    W: torch.nn.Linear
    alpha: float
    target_features: list of [n, d_out]
    original_features: list of [n, d_in]
    regularization: list of torch.Tensor (each [m, d_in]) 或 None
    """
    device = W.weight.device
    X = torch.cat(original_features, dim=0).to(device)   # [N, d_in]
    Y_raw = torch.cat(target_features, dim=0).to(device)  # [N, d_in] 或 [N, d_in_target]
    # 将 Y 变成 Y @ W^T （加 bias 可选）
    Y = Y_raw @ W.weight.T
    # 当前权重
    W_old = W.weight.data.clone()  # [d_out, d_in]

    # 基础部分 XtX, XtY
    XtX = X.T @ X       # [d_in, d_in]
    XtY = X.T @ Y       # [d_in, d_out]

    if regularization is not None:
        # 把所有 reg 向量拼起来
        R = torch.cat(regularization, dim=0).to(device)  # [M, d_in]

        # 惩罚项： (WR^T - W_old R^T)^2
        # 展开后等价于在 normal equation 里加上 α (R^T R) 和 α (R^T R W_old^T)
        RtR = R.T @ R  # [d_in, d_in]
        RtWoldT = R.T @ (R @ W_old.T)  # [d_in, d_out]

        A = XtX + alpha * RtR
        B = XtY + alpha * RtWoldT
    else:
        # 普通 L2 正则化
        A = XtX + alpha * torch.eye(X.shape[1], device=device)
        B = XtY

    # 解 A W^T = B
    W_new = torch.linalg.solve(A, B)  # [d_in, d_out]
    W.weight.data = W_new.T.contiguous()

def ensure_row_features(tensors):
    """
    检查并转化输入特征张量为 [n, d] 形式 (行向量为样本)。
    输入: list[Tensor] 或 Tensor
    输出: list[Tensor]
    """
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]

    out = []
    for t in tensors:
        if t.ndim == 1:
            # 一维向量 [d] -> [1, d]
            t = t.unsqueeze(0)
        elif t.ndim == 2:
            n, d = t.shape
            # 如果是 [d, 1] 可能是列向量，把它转成 [1, d]
            if n == 1 and d > 1:
                pass  # 已经是行向量
            elif d == 1 and n > 1:
                # 列向量 -> 行向量 (转置)
                t = t.T
        else:
            raise ValueError(f"特征维度必须是 1D 或 2D 张量，收到 {t.shape}")

        out.append(t)
    return out

from transformers import AutoTokenizer, PretrainedConfig

def get_text_embedding(tokenizer, text_encoder, prompts, device):
    text_input = tokenizer(
        prompts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    )
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    return text_embeddings

def extract_cross_attention_kv(unet):
    kv_dict = {}
    for name, module in unet.named_modules():
        # Cross-Attention 一般在 attn2
        if "attn2" in name and hasattr(module, "to_k") and hasattr(module, "to_v"):
            kv_dict[name + ".to_k"] = module.to_k.weight.detach().clone()
            kv_dict[name + ".to_v"] = module.to_v.weight.detach().clone()
    return kv_dict

def assign_cross_attention_kv(unet, kv_dict):
    with torch.no_grad():
        for name, module in unet.named_modules():
            if "attn2" in name and hasattr(module, "to_k") and hasattr(module, "to_v"):
                if name + ".to_k" in kv_dict:
                    module.to_k.weight.copy_(kv_dict[name + ".to_k"])
                if name + ".to_v" in kv_dict:
                    module.to_v.weight.copy_(kv_dict[name + ".to_v"])
    print("所有 Cross-Attn 的 Wk, Wv 已更新！")

def extract_cross_attention_Wo(unet):
    Wo_dict = {}
    for name, module in unet.named_modules():
        if "attn2" in name and hasattr(module, "to_out") and hasattr(module.to_out, "weight"):
            Wo_dict[name + ".to_out.weight"] = module.to_out.weight.detach().clone()
    return Wo_dict

def assign_cross_attention_Wo(unet, Wo_dict):
    with torch.no_grad():
        for name, module in unet.named_modules():
            if "attn2" in name and hasattr(module, "to_out") and hasattr(module.to_out, "weight"):
                if name + ".to_out.weight" in Wo_dict:
                    module.to_out.weight.copy_(Wo_dict[name + ".to_out.weight"])
    print("所有 Cross-Attn 的 Wo 已更新！")

def extract_cross_attention_weights(unet):
    """
    提取 unet 中所有 cross-attention (attn2) 层的 Wk, Wv, Wo
    返回字典: { layer_name: {"Wk": tensor, "Wv": tensor, "Wo": tensor} }
    """
    attn_weights = {}
    for name, module in unet.named_modules():
        if "attn2" in name:  # Cross-Attention 层
            entry = {}
            if hasattr(module, "to_q"):
                entry["Wq"] = module.to_q.weight.detach().clone()
            if hasattr(module, "to_k"):
                entry["Wk"] = module.to_k.weight.detach().clone()
            if hasattr(module, "to_v"):
                entry["Wv"] = module.to_v.weight.detach().clone()
            if hasattr(module, "to_out") and hasattr(module.to_out, "0"):  # to_out 是 Sequential
                entry["Wo"] = module.to_out[0].weight.detach().clone()
            if entry:
                attn_weights[name] = entry
    return attn_weights

def assign_cross_attention_weights(unet, attn_weights):
    """
    将修改后的权重批量赋值回 unet
    attn_weights: 与 extract_cross_attention_weights 相同结构的字典
    """
    for name, module in unet.named_modules():
        if name in attn_weights:
            entry = attn_weights[name]
            if "Wk" in entry and hasattr(module, "to_k"):
                module.to_k.weight.data.copy_(entry["Wk"])
            if "Wv" in entry and hasattr(module, "to_v"):
                module.to_v.weight.data.copy_(entry["Wv"])
            if "Wo" in entry and hasattr(module, "to_out") and hasattr(module.to_out, "0"):
                module.to_out[0].weight.data.copy_(entry["Wo"])
    print("所有 Cross-Attn 的 Wk, Wv, Wo 已更新！")

import torch
import torch.nn.functional as F

def manual_cross_attention(latent, text_embedding, attn_weight_entry):
    """
    手动计算 Cross-Attention
    Args:
        latent: [B, Lq, d_model] 查询向量
        text_embedding: [B, Lk, d_model_text] 文本 embedding
        attn_weight_entry: dict, 包含 "Wk", "Wv", "Wo" 三个矩阵
    Returns:
        output: [B, Lq, d_model] Cross-Attention 输出
    """
    Wk = attn_weight_entry["Wk"]  # [d_model_text, H*head_dim]
    Wv = attn_weight_entry["Wv"]  # [d_model_text, H*head_dim]
    Wo = attn_weight_entry["Wo"]  # [H*head_dim, d_model]
    Wq = attn_weight_entry["Wq"] if "Wq" in attn_weight_entry else None  # [d_model, H*head_dim] 可选
    # 1. 计算 K, V
    # text_embedding @ Wk/Wv
    # text_embedding: [B, Lk, d_model_text]
    # K,V: [B, Lk, H*head_dim]
    K = text_embedding @ Wk
    V = text_embedding @ Wv

    # 2. 计算 Q (latent -> Q)，通常 Q 使用自己的 Wq，但这里我们 focus K/V/Wo
    if Wq is not None:
        Q = latent @ Wq  # [B, Lq, H*head_dim]
    else:
        Q = latent  # 简化为 identity 映射，[B,Lq,d_model]

    # 3. 计算注意力得分
    d_head = K.shape[-1]  # 多头已经拼接好了
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_head ** 0.5)  # [B, Lq, Lk]

    # 4. softmax
    attn_probs = F.softmax(attn_scores, dim=-1)  # [B, Lq, Lk]

    # 5. 加权 V
    context = torch.matmul(attn_probs, V)  # [B, Lq, H*head_dim]

    # 6. 输出投影 Wo
    output = context @ Wo  # [B, Lq, d_model]

    return output

def minimal_distance(unsafe_matrix, prompt):
    """
    计算最小距离 ||p - Cz||_2，其中 z = (C^T C)^{-1} C^T p
    unsafe_matrix: torch.Tensor, shape [n, m]
    prompt: torch.Tensor, shape [n]
    """
    C = unsafe_matrix
    p = prompt
    Ct = C.t()
    z = torch.linalg.solve(Ct @ C, Ct @ p)
    diff = p - C @ z
    return torch.norm(diff)

