import torch
import triton
from .kernels import _gradient_matmul, _linear_matmul, _bwd_kernel1, _bwd_kernel2, _bwd_kernel3, _bwd_kernel4

class SpatialGradFunction(torch.autograd.Function):

  @staticmethod
  def forward(ctx, x, gr, gi, lr, li):
    m, k = gr.shape
    _, n = x.shape
    xgr = torch.empty((m, n), dtype=x.dtype, device=x.device)
    xgi = torch.empty((m, n), dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_M']), triton.cdiv(n, meta['BLOCK_N']))
    _gradient_matmul[grid](gr, gi, x, xgr, xgi,
                          m, n, k,
                          gr.stride(0), gr.stride(1), gi.stride(0), gi.stride(1), x.stride(0), x.stride(1),
                          xgr.stride(0), xgr.stride(1), xgi.stride(0), xgi.stride(1),
                          BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                          GROUP_SIZE=8)

    z = torch.empty((m, n), dtype=xgr.dtype, device=xgr.device)
    _linear_matmul[grid](xgr, xgi, lr, li, z,
                        m, n, k,
                        xgr.stride(0), xgr.stride(1), xgi.stride(0), xgi.stride(1), lr.stride(0), lr.stride(1), li.stride(0), li.stride(1),
                        z.stride(0), z.stride(1),
                        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                        GROUP_SIZE=8)

    ctx.save_for_backward(gr, gi, lr, li, xgr, xgi, z)
    return z

  @staticmethod
  def backward(ctx, dZ):
    dX, dGR, dGI, dLR, dLI = None, None, None, None, None
    gr, gi, lr, li, xgr, xgi, z = ctx.saved_tensors

    m, k = xgr.shape
    n, _ = lr.shape
    grad_x_grads_real = torch.empty((m, n), dtype=dZ.dtype, device=dZ.device)
    grad_x_grads_im = torch.empty((m, n), dtype=dZ.dtype, device=dZ.device)
    grad_x_real = torch.empty((m, n), dtype=dZ.dtype, device=dZ.device)
    grad_x_im = torch.empty((m, n), dtype=dZ.dtype, device=dZ.device)
    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_M']), triton.cdiv(n, meta['BLOCK_N']))
    _bwd_kernel1[grid](z, dZ, xgr, xgi, lr, li,
                        grad_x_grads_real, grad_x_grads_im, grad_x_real, grad_x_im,
                        m, n, k,
                        z.stride(0), z.stride(1), dZ.stride(0), dZ.stride(1), xgr.stride(0), xgr.stride(1), xgi.stride(0), xgi.stride(1), lr.stride(0), lr.stride(1), li.stride(0), li.stride(1),
                        grad_x_grads_real.stride(0), grad_x_grads_real.stride(1), grad_x_grads_im.stride(0), grad_x_grads_im.stride(1), grad_x_real.stride(0), grad_x_real.stride(1), grad_x_im.stride(0), grad_x_im.stride(1),
                        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                        GROUP_SIZE=8)
    
    # linear weights
    dLR = torch.empty((k, n), dtype=dZ.dtype, device=dZ.device)
    dLI = torch.empty((k, n), dtype=dZ.dtype, device=dZ.device)
    grid2 = lambda meta: (triton.cdiv(k, meta['BLOCK_M']), triton.cdiv(n, meta['BLOCK_N']))
    _bwd_kernel2[grid2](xgr, xgi, grad_x_real, grad_x_im,
                       dLR, dLI,
                       k, n, m,
                       xgr.stride(0), xgr.stride(1), xgi.stride(0), xgi.stride(1), grad_x_real.stride(0), grad_x_real.stride(1), grad_x_im.stride(0), grad_x_im.stride(1),
                       dLR.stride(0), dLR.stride(1), dLI.stride(0), dLI.stride(1),
                       BLOCK_M=32, BLOCK_N=32, BLOCK_K=128,
                       GROUP_SIZE=8)

    # x
    grad_x_grads_real_total = torch.empty((m, n), dtype=xgr.dtype, device=xgr.device)
    grad_x_grads_im_total = torch.empty((m, n), dtype=xgr.dtype, device=xgr.device)
    _bwd_kernel3[grid](grad_x_real, grad_x_im, lr, li, grad_x_grads_real, grad_x_grads_im,
                       grad_x_grads_real_total, grad_x_grads_im_total,
                       m, n, k,
                       grad_x_real.stride(0), grad_x_real.stride(1), grad_x_im.stride(0), grad_x_im.stride(1), lr.stride(0), lr.stride(1), li.stride(0), li.stride(1),
                       grad_x_grads_real.stride(0), grad_x_grads_real.stride(1), grad_x_grads_im.stride(0), grad_x_grads_im.stride(1),
                       grad_x_grads_real_total.stride(0), grad_x_grads_real_total.stride(1), grad_x_grads_im_total.stride(1), grad_x_grads_im_total.stride(1),
                       BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                       GROUP_SIZE=8)

    k, m = gr.shape
    dX = torch.empty((m, n), dtype=dZ.dtype, device=dZ.device)
    _bwd_kernel4[grid](gr, gi, grad_x_grads_real_total, grad_x_grads_im_total,
                       dX,
                       m, n, k,
                       gr.stride(0), gr.stride(1), gi.stride(0), gr.stride(1),
                       grad_x_grads_real_total.stride(0), grad_x_grads_real_total.stride(1), grad_x_grads_im_total.stride(0), grad_x_grads_im_total.stride(1),
                       dX.stride(0), dX.stride(1),
                       BLOCK_M=32, BLOCK_N=32, BLOCK_K=128,
                       GROUP_SIZE=8)

    return dX, dGR, dGI, dLR, dLI