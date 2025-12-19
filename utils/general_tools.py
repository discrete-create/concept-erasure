import torch

def is_within_cos_tolerance(tensor1, tensor2, alpha):
    """
    判断两个张量的余弦距离是否在最大容差alpha内
    返回：True（在容差内），False（超出容差）
    """
    cos_sim = torch.nn.functional.cosine_similarity(tensor1.flatten(), tensor2.flatten(), dim=0)
    cos_dist = 1 - cos_sim
    return cos_dist <= alpha

def is_within_L2_tolerance(tensor1, tensor2, alpha):
    """
    判断两个张量的L2距离是否在最大容差alpha内
    返回：True（在容差内），False（超出容差）
    """
    l2_dist = torch.norm(tensor1 - tensor2)
    return l2_dist <= alpha

