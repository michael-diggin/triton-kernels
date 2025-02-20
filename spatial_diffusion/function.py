import torch
import triton
from .kernels import _dd_ds_bwd, _dx_bwd, _splitk_diffusion


class FlashDiffusionSplitK(torch.autograd.Function):

  @staticmethod
  def forward(ctx, b, x, mass, e, d):
    k, m = b.shape
    _, n = x.shape
    s = torch.empty((m, n), dtype=x.dtype, device=x.device)
    b_t = b.transpose(-2, -1)

    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_M']), triton.cdiv(n, meta['BLOCK_N']), meta["SPLIT_K"])

    _splitk_diffusion[grid](b_t, x, mass, e, d, s,
                          m, n, k,
                          b_t.stride(0), b_t.stride(1), x.stride(0), x.stride(1),
                          s.stride(0), s.stride(1),
                          BLOCK_M=32, BLOCK_N=64, BLOCK_K=64,
                          GROUP_SIZE=4, SPLIT_K=6,
                          num_warps=4, num_stages=4)
    output = torch.matmul(b, s)
    ctx.save_for_backward(s, b, mass, e, d)
    return output, s


  @staticmethod
  def backward(ctx, dO, dS):
    s, b, mass, e, d = ctx.saved_tensors
    dD, dX = None, None

    m, n = dS.shape
    v, _ = dO.shape

    # first step - compute s_grad = dS + b.T@dO and dD in one kernel
    b_t = b.transpose(-2, -1)
    dD = torch.empty((n), dtype=dS.dtype, device=dS.device)
    dst = torch.empty((m, n), dtype=dS.dtype, device=dS.device)
    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_M']), triton.cdiv(n, meta['BLOCK_N']), meta["SPLIT_K"])
    split_k = 6 if v > 20000 else 1 # heuristic
    _dd_ds_bwd[grid](dS, dO, b_t, s, e, d,
                    dD, dst,
                    m, n, v,
                    dS.stride(0), dS.stride(1), dO.stride(0), dO.stride(1), b_t.stride(0), b_t.stride(1), s.stride(0), s.stride(1),
                    dst.stride(0), dst.stride(1),
                    BLOCK_M=128, BLOCK_N=16, BLOCK_K=64,
                    GROUP_SIZE=8, SPLIT_K=split_k,
                    num_warps=4, num_stages=4)
    if ctx.needs_input_grad[1]:
      # the input X doesn't always need grads, eg first layer
      #dX = torch.matmul(b*mass.unsqueeze(-1), dst)

      dX = torch.empty((v, n), dtype=dS.dtype, device=dS.device)
      # this is where I've overloaded M, N, K with V, N, M
      grid = lambda meta: ((triton.cdiv(v, meta['BLOCK_M']), triton.cdiv(n, meta['BLOCK_N'])))
      dx_bwd[grid](b, dst, mass,
                    dX,
                    v, n, m,
                    b.stride(0), b.stride(1), dst.stride(0), dst.stride(1),
                    dX.stride(0), dX.stride(1),
                    BLOCK_M=32, BLOCK_N=32, BLOCK_K=128,
                    GROUP_SIZE=8,
                    num_warps=4, num_stages=4)


    return None, dX, None, None, dD

splitk_diffusion = FlashDiffusionSplitK.apply