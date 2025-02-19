import torch
import triton
from .kernels import fused_diffusion, dd_ds_bwd, dx_bwd, _splitk_diffusion


class FlashDiffusion(torch.autograd.Function):
  '''
  A custom function that can handle the forward pass and 
  backward pass for the fused_diffusion kernel
  '''

  @staticmethod
  def forward(ctx, basis, x, mass, e, d):
    '''
    Returns O and S
    Normal Output and Spectral output
    '''
    k, m = basis.shape
    _, n = x.shape
    spectral = torch.empty((m, n), dtype=x.dtype, device=x.device)
    b_t = basis.transpose(-2, -1)

    # launch the triton kernel with specific meta parameters
    # these are a bit magic, but autotuning was used to determine them with benchmarks
    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_M']), triton.cdiv(n, meta['BLOCK_N']))
    fused_diffusion[grid](b_t, x, mass, e, d, spectral,
                          m, n, k,
                          b_t.stride(0), b_t.stride(1), x.stride(0), x.stride(1),
                          spectral.stride(0), spectral.stride(1),
                          BLOCK_M=32, BLOCK_N=32, BLOCK_K=64,
                          GROUP_SIZE=8,
                          num_warps=4, num_stages=4)
    output = torch.matmul(basis, spectral)
    # save some of the inputs and an output for the backward pass
    # no intermediate tensors are needed
    ctx.save_for_backward(spectral, basis, mass, e, d)
    return output, spectral
    
  @staticmethod
  def backward(ctx, dO, dS):
    # Custom backward implementation
    # use the tensors saved from the forward pass
    # to recompute the tensors that are needed

    # Gradients need to be accumulated from both dS and from dO
    # since O = b@S, dO flows right through to dS
    # just use dS instead:
    # so dS = dS + b.T@dO

    # dD = (-e.unsqueeze(-1)*dS*S).sum(dim=0)
    # dX = (mass.unsqueeze(-1)*b)@(dS*exp(-e.unsqueeze(-1)*d))

    s, b, mass, e, d = ctx.saved_tensors
    dD, dX = None, None

    m, n = dS.shape
    v, _ = dO.shape

    # first step - compute dD and dst = (dS + b.T@dO)*(exp(-e*d)) in one Kernel
    # because dD involves a sum along axis=0, and that dimension always has size 128
    # use 128 as BLOCK_M, this allows the kernel to directly reduce along that axis
    b_t = b.transpose(-2, -1)
    dD = torch.empty((n), dtype=dS.dtype, device=dS.device)
    dst = torch.empty((m, n), dtype=dS.dtype, device=dS.device)

    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_M']), triton.cdiv(n, meta['BLOCK_N']))
    dd_ds_bwd[grid](dS, dO, b_t, s, e, d,
                    dD, dst,
                    m, n, v,
                    dS.stride(0), dS.stride(1), dO.stride(0), dO.stride(1), b_t.stride(0), b_t.stride(1), s.stride(0), s.stride(1),
                    dst.stride(0), dst.stride(1),
                    BLOCK_M=128, BLOCK_N=16, BLOCK_K=64,
                    GROUP_SIZE=8,
                    num_warps=4, num_stages=4)
    if ctx.needs_input_grad[1]:
      # the input X doesn't always need grads, eg first layer

      # second kernel computes dX using the intermediate dst
      # fusing the matrix multiply with the elementwise vector*matrix product
      dX = torch.empty((v, n), dtype=dS.dtype, device=dS.device)
      grid = lambda meta: ((triton.cdiv(v, meta['BLOCK_M']), triton.cdiv(n, meta['BLOCK_N'])))
      dx_bwd[grid](b, dst, mass,
                  dX,
                  v, n, m,
                  b.stride(0), b.stride(1), dst.stride(0), dst.stride(1),
                  dX.stride(0), dX.stride(1),
                  BLOCK_M=32, BLOCK_N=32, BLOCK_K=128,
                  GROUP_SIZE=8,
                  num_warps=4, num_stages=4,
                  )
    return None, dX, None, None, dD

# the function that callers can import and use
diffusion_func = FlashDiffusion.apply


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
    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_M']), triton.cdiv(n, meta['BLOCK_N']))
    dd_ds_bwd[grid](dS, dO, b_t, s, e, d,
                    dD, dst,
                    m, n, v,
                    dS.stride(0), dS.stride(1), dO.stride(0), dO.stride(1), b_t.stride(0), b_t.stride(1), s.stride(0), s.stride(1),
                    dst.stride(0), dst.stride(1),
                    BLOCK_M=128, BLOCK_N=16, BLOCK_K=64,
                    GROUP_SIZE=8,
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