import triton
import triton.language as tl
import torch

@triton.jit
def _second_kernel(A, B, C, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr,
            ):
    # matrix multiplication

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
    tl.atomic_add(C, acc)

def second_kernel(x, y):
  m, k = x.shape
  _, n = y.shape
  z = torch.zeros((m, n), dtype=x.dtype, device=x.device) # needs to be zeros
  grid = lambda meta: (triton.cdiv(m, meta['BLOCK_M']), triton.cdiv(n, meta['BLOCK_N']), meta["SPLIT_K"])
  _second_kernel[grid](x, y, z,
                       m, n, k,
                       x.stride(0), x.stride(1), y.stride(0), y.stride(1), z.stride(0), z.stride(1),
                       BLOCK_M=32, BLOCK_N=64, BLOCK_K=32, SPLIT_K=4,
                       num_warps=4, num_stages=4,
                       GROUP_M=8)
  return z
