import torch
import triton
import triton.language as tl
import time
import gc
import torch.nn.functional as F
from torch import nn
@triton.jit
def fp16A_int8B_gemm_kernel(
    A_ptr, Bq_t_ptr, S_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_s,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    Bq_t_ptrs = Bq_t_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    S_ptrs = S_ptr + offs_k[:, None] * stride_s

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        A = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0).to(tl.float32)
        Bq_t = tl.load(Bq_t_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0).to(tl.int8)
        S = tl.load(S_ptrs, mask=(offs_k[:, None] < K), other=1.0).to(tl.float32)

        B = Bq_t.to(tl.float32) * S

        acc += tl.dot(A, B)

        A_ptrs += BLOCK_K * stride_ak
        Bq_t_ptrs += BLOCK_K * stride_bk
        S_ptrs += BLOCK_K * stride_s

    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def fp16A_int8B_gemm(A, Bq_t, S, BLOCK=(64,64,32)):
    #print("A's shape:", A.shape)
    #print("Bq_t's shape:", Bq_t.shape)
    #Bq_t = Bq_t.contiguous()
    N = Bq_t.shape[1]
    is_3d = (A.ndim == 3)
    if is_3d:
        B_dim, L, K = A.shape
        A_2d = A.reshape(-1, K)   # [B*L, K]
        M = B_dim * L
    elif A.ndim == 2:
        M, K = A.shape
        A_2d = A.contiguous()
    else:
        raise ValueError(f"A must be 2D or 3D, got {A.ndim}D")
    #print("A's dtype is", A.dtype)
    #print("B_q's dtype is", B_q.dtype)
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)
    S= S.contiguous().to(torch.float32)
    grid = (triton.cdiv(M, BLOCK[0]), triton.cdiv(N, BLOCK[1]))

    fp16A_int8B_gemm_kernel[grid](
        A_2d, Bq_t, S, C,
        M, N, K,
        A_2d.stride(0), A_2d.stride(1),
        Bq_t.stride(0), Bq_t.stride(1),
        S.stride(0),
        C.stride(0), C.stride(1),
        BLOCK[0], BLOCK[1], BLOCK[2]
    )
    if is_3d:
        C = C.reshape(B_dim, L, N)
        return C
    
    return C

def test_fp16A_int8B(M=16, N=8192, K=2560, device='cuda'):
    torch.manual_seed(0)
    print("test size is",M,K,N)
    A = torch.randn(M, K, device=device, dtype=torch.float16)
    B = torch.randn(K, N, device=device, dtype=torch.float16)
    # Quantize B to int8 with per-row scales
    S = B.abs().max(dim=1).values / 127.0  #
    Bq = (B / S[:, None]).round().clamp(-127, 127).to(torch.int8).contiguous()
    S = S.contiguous()

    # Triton kernel
    # warmup
    torch.cuda.synchronize()
    start = time.time()
    C_triton = fp16A_int8B_gemm(A, Bq, S, BLOCK=(64,64,64))
    torch.cuda.synchronize()
    warmup_time = time.time() - start
    #print(f"Triton kernel warmup time: {warmup_time*1000:.2f} ms")
    #print(A.device,Bq.device,S.device)
    # timed run
    iter=1000
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iter):
        C_triton = fp16A_int8B_gemm(A, Bq, S, BLOCK=(64,64,32))
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / iter
    print(f"Triton kernel average time over {iter} runs: {triton_time*1000:.2f} ms")


    # PyTorch reference
    # warmup
    torch.cuda.synchronize()
    start = time.time()
    B_deq = (Bq.to(torch.float32)* S[:, None])
    C_ref = torch.matmul(A.to(torch.float32), B_deq).to(torch.float16)
    torch.cuda.synchronize()
    warmup_time = time.time() - start
    #print(f"PyTorch reference warmup time: {warmup_time*1000:.2f} ms")
    #print(A.device,Bq.device,S.device)
    # timed run
    iter=1000
    torch.cuda.synchronize()
    start = time.time()
    S=S.to(torch.float16)
    for _ in range(iter):
        C_ref = torch.matmul(A.to(torch.float16), Bq.to(torch.float16) * S[:, None]).to(torch.float16)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / iter
    print(f"PyTorch reference average time over {iter} runs: {torch_time*1000:.2f} ms")
    print(f"Speedup: {torch_time/triton_time:.2f}x")
    max_diff = (C_triton - C_ref).abs().max().item()
    mean_diff = (C_triton - C_ref).abs().mean().item()
    print(f"Mean difference: {mean_diff}")
    print(f"Max difference: {max_diff}")
    #assert max_diff < 1e-2, f"Max difference {max_diff} exceeds tolerance"

def quantize_weight_per_out_channel(W: torch.Tensor):
    W_f = W.float()
    max_per_col = W_f.abs().amax(dim=0, keepdim=True)
    scale = (max_per_col / 127.0)
    scale[scale == 0] = 1e-8
    W_q = torch.round(W_f / scale).clamp(-127,127).to(torch.int8)
    return W_q.t().contiguous(), scale.view(-1).to(torch.float16)

def quantize_activation_per_tensor(A: torch.Tensor):
    A_f = A.float()
    max_val = A_f.abs().amax()
    scale = (max_val/127.0).clamp(min=1e-8)
    A_q = torch.round(A_f/scale).clamp(-127,127).to(torch.int8)
    return A_q, torch.tensor(float(scale), dtype=torch.float16, device=A.device)

# --------------------------
# Example run
# --------------------------
class QuantLinearTriton(nn.Linear):
    def __init__(self, in_features, out_features, name,bias=True, block=(64,64,32)):
        super().__init__(in_features, out_features, bias=bias)
        self.block = block
        self.name = name
        # buffers (设置为 None，quantize_weight 会填充)
        self.register_buffer("weight_q", None)   # int8 tensor (out, in)
        self.register_buffer("scale", None)      # float16 or float32 (out,1)
    def quantize_weight(self, W: torch.Tensor):
        W_q, scale = quantize_weight_per_out_channel(W)
        self.weight_q = W_q
        self.scale = scale.contiguous()
    def forward(self, x: torch.Tensor):
        assert self.weight_q is not None, "Weight not quantized yet. Call quantize_weight first."
        #print("Input x shape:", x.shape)
        #print("weight_q shape:", self.weight_q.shape)
        #print("layer name",self.name)
        #if self.name=="q_proj" or self.name=="k_proj" or self.name=="v_proj" or self.name=="o_proj" or self.name=="up_proj" or self.name=="down_proj":
        #print("Using triton kernel for",self.name)
        #print("X.size:",x.size())
        #print("weight_q.size:",self.weight_q.size())
        C = fp16A_int8B_gemm(x, self.weight_q, self.scale, BLOCK=self.block)
        #else:
        #    C = F.linear(x.float(), (self.weight_q.float()*self.scale.view(-1,1).float()).t(), bias=None)
        if self.bias is not None:
            C = C + self.bias.view(1,-1)
        #print("Output C shape:", C.shape)   
        return C



def replace_linear(model: nn.Module, mode="int8_per_channel", group_size=128, exclude=None):
    """
    Replace nn.Linear layers with QuantLinearTriton layers in the given model.
    mode: "int8_per_channel" only supported for now.
    exclude: list of layer names to exclude from replacement.
    """
    for name, module in model.named_children():
        if exclude is not None and name in exclude:
            continue
        if isinstance(module, nn.Linear) :#and name in ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj"]:
            qlinear = QuantLinearTriton(
                in_features=module.in_features,
                out_features=module.out_features,
                name=name,
                bias=(module.bias is not None),
                block=(64,64,32)
            ).to(next(model.parameters()).device)
            qlinear.quantize_weight(module.weight.data)
            if module.bias is not None:
                qlinear.bias.data = module.bias.data.half()
            setattr(model, name, qlinear)
            del module
            torch.cuda.empty_cache()
            gc.collect()
        else:
            replace_linear(module, mode=mode, group_size=group_size, exclude=exclude)


#test_fp16A_int8B(5, 8192, 2560, device='cuda')
#test_fp16A_int8B(1, 4096, 2560, device='cuda')
#test_fp16A_int8B(5, 1024,2560, device='cuda')
#test_fp16A_int8B(5, 8192, 2560, device='cuda')
#test_fp16A_int8B(1, 9728, 2560, device='cuda')
#test_fp16A_int8B(5, 2560, 9728, device='cuda')
#test_fp16A_int8B(3, 2560, 9728, device='cuda')
