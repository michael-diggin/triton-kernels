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
    # matrix multiplication
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    incr = BLOCK_K*SPLIT_K
    for k in range(0, tl.cdiv(K, incr)):
        k_remaining = K - k * (incr)
        a = tl.load(A, mask=rk[None, :] < k_remaining, other=0.0)
        b = tl.load(B, mask=rk[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b)
        A += incr * stride_ak
        B += incr * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
      tl.atomic_add(C, acc, mask=mask)


def splitk_kernel(x, y):
    """
    Executes the _splitk_kernel using the split K algorithm
    the output matrix must be provided in float16 as atomic adds do not
    support bfloat16.
    We also set the output to be zero rather than empty due to the atomic add
    """
    m, k = x.shape
    _, n = y.shape
    z = torch.zeros((m, n), dtype=torch.float16, device=x.device)
    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_M'])*triton.cdiv(n, meta['BLOCK_N']), meta["SPLIT_K"])
    _splitk_kernel[grid](x, y, z,
                         m, n, k,
                         x.stride(0), x.stride(1), y.stride(0), y.stride(1), z.stride(0), z.stride(1),
                         BLOCK_M=32, BLOCK_N=32, BLOCK_K=128, SPLIT_K=8,
                         num_warps=4, num_stages=2,
                         GROUP_M=8)
    return z

def normal_kernel(x, y):
    """
    Executes the _splitk_kernel with split K == 1
    This executes it 'as normal', ie a simple block tiled algorithm
    """
    m, k = x.shape
    _, n = y.shape
    z = torch.empty((m, n), dtype=torch.float16, device=x.device)
    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_M'])*triton.cdiv(n, meta['BLOCK_N']), meta["SPLIT_K"])
    _splitk_kernel[grid](x, y, z,
                         m, n, k,
                         x.stride(0), x.stride(1), y.stride(0), y.stride(1), z.stride(0), z.stride(1),
                         BLOCK_M=32, BLOCK_N=32, BLOCK_K=256, SPLIT_K=1,
                         num_warps=4, num_stages=6,
                         GROUP_M=8)
    return z