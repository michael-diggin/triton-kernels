import torch
import triton
import triton.language as tl


@triton.jit
def _splitk_kernel(A, B, C, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr,
            ):
    
    """
    Performs matrix multiplication using the split K algorithm.
    C = A @ B

    A: (M, K) input matrix
    B: (K, N) input matrix
    C: (M, N) output matrix
    """
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(1), tl.num_programs(2)
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_M)
    pid_k = tl.program_id(2)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_K), BLOCK_K)
    # pointers
    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K*SPLIT_K)):
        k_remaining = K - k * (BLOCK_K*SPLIT_K)
        a = tl.load(A, mask=rk[None, :] < k_remaining, other=0.0)
        b = tl.load(B, mask=rk[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b)
        A += BLOCK_K*SPLIT_K * stride_ak
        B += BLOCK_K*SPLIT_K * stride_bk

    acc = acc.to(C.dtype.element_ty)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.atomic_add(C, acc, sem="relaxed")

def _zero_output(*args, **kwargs):
    if kwargs["SPLIT_K"] != 1:
      args[2].zero_()


_splitk_kernel.add_pre_run_hook(_zero_output)


def splitk_kernel(x, y):
    """
    Executes the _splitk_kernel using the split K algorithm
    """
    m, k = x.shape
    _, n = y.shape
    z_dtype = torch.float32
    if x.dtype == torch.float16 and y.dtype == torch.float16:
        z_dtype = torch.float16
    z = torch.empty((m, n), dtype=z_dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_M']), triton.cdiv(n, meta['BLOCK_N']), meta["SPLIT_K"], )
    _splitk_kernel[grid](x, y, z,
                         m, n, k,
                         x.stride(0), x.stride(1), y.stride(0), y.stride(1), z.stride(0), z.stride(1),
                         BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, SPLIT_K=6,
                         num_warps=4, num_stages=4,
                         GROUP_M=4)
    return z
