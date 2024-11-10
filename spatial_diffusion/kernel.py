import torch
import triton
import triton.language as tl


def extra_configs():
  """
  List of extra configs I tried
  It seemed to just stick with (32, 32, 64, 4, 4)
  """
  return [
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 256}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 256}, num_stages=4, num_warps=4),
  ]

# Additional configs can be added here, which Triton will try out to find the fastest one
# if any of the inputs in 'key' change
# eg
# @triton.autotune(extra_configs(), key=["M", "N", "K"])
@triton.jit
def fused_diffusion(
    # pointers to inputs and output
    b_ptr, x_ptr, m_ptr, e_ptr, d_ptr, z_ptr,
    # Matrix dimensions
    M, N, K,
    # matrix strides
    stride_bm, stride_bk, stride_xk, stride_xn, 
    stride_zm, stride_zn,
    # block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    # meta params
    GROUP_SIZE: tl.constexpr
):
    
    # B is a [M, K] matrix (input)
    # X is a [K, N] matrix (input)
    # M is a [K] row vector (input)
    # E is a [M] row vector (inuput)
    # D is a [N] row vector (input)
    # Z is a [M, N] matrix (output)

    # Z = exp(-E.unsqueeze(-1)*D)*(B@(X*M.unsqueeze(-1)))

    # Map program ids `pid` to the block of Z it should compute.
    # This is done in a swizzled/grouped ordering to promote L2 data reuse.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)
    pid_0, pid_1 = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE)

    # Offsets for the K dimension (looped dimension)
    offs_k = tl.arange(0, BLOCK_K)  # range within the K block

    # Compute row and column indices for the output Z block
    z_row_indices = pid_0 * BLOCK_M + tl.arange(0, BLOCK_M)  # Output rows in Z
    z_col_indices = pid_1 * BLOCK_N + tl.arange(0, BLOCK_N)  # Output cols in Z

    # hint to the compiler that these are contiguous
    offs_m = tl.max_contiguous(tl.multiple_of(z_row_indices, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(z_col_indices, BLOCK_N), BLOCK_N)
    offs_k_c = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    # set up the pointers to the first blocks that get loaded
    # the stride values are in case the tensors aren't contiguous in memory
    b_ptrs = b_ptr + offs_m[:, None]*stride_bm + offs_k_c[None, :]*stride_bk
    x_ptrs = x_ptr + offs_k_c[:, None]*stride_xk + offs_n[None, :]*stride_xn
    m_ptrs = m_ptr + offs_k_c[:, None]

    b_mask = (z_row_indices < M)[:, None]
    x_mask = (z_col_indices < N)[None, :]

    # Initialize accumulator for this (BLOCK_M, BLOCK_N) block in the output
    # This is in float32 to avoid losing precision
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    lim = K
    b_k = BLOCK_K*stride_bk
    x_k = BLOCK_K*stride_xk

    # Loop over the K dimension in chunks of BLOCK_K
    for _ in range(0, tl.cdiv(K, BLOCK_K)):

      b_block = tl.load(b_ptrs, mask=(offs_k[None, :] < lim) & b_mask, other=0.0)
      x_block = tl.load(x_ptrs, mask=(offs_k[:, None] < lim) & x_mask, other=0.0)
      m_block = tl.load(m_ptrs, mask=(offs_k[:, None] < lim), other=0.0)

      # Accumulate the result of the dot product
      accumulator += tl.dot(b_block, x_block*m_block)

      b_ptrs += b_k
      x_ptrs += x_k
      m_ptrs += BLOCK_K
      lim -= BLOCK_K

    accumulator = accumulator.to(x_ptr.dtype.element_ty)

    # second step - load E and D block
    e_block = tl.load(e_ptr+offs_m, mask=z_row_indices<M, other=0.0).expand_dims(-1)
    d_block = tl.load(d_ptr+offs_n, mask=z_col_indices<N, other=0.0).expand_dims(0)

    # make use of the following identity - supposedly 'more stable'
    # tl.exp(x) = tl.exp2(log2(e)*x)
    minus_log2_e = -1.4426950408889634
    diff_coefs = tl.math.exp2(minus_log2_e*e_block*d_block)
    accumulator = diff_coefs*accumulator

    # Set up pointers and masks for storing the output block in Z
    z_ptrs = z_ptr + offs_m[:, None]*stride_zm + offs_n[None, :]*stride_zn
    z_mask = (z_row_indices < M)[:, None] & (z_col_indices < N)[None, :]

    # Store the accumulated result in Z
    tl.store(z_ptrs, accumulator, mask=z_mask)


class FlashDiffusion(torch.autograd.Function):
  '''
  A custom function that can handle the forward pass and 
  backward pass for the fused_diffusion kernel
  '''

  @staticmethod
  def forward(ctx, b, x, mass, e, d):
    '''
    Returns O and S
    Normal Output and Spectral output
    '''
    k, m = b.shape
    _, n = x.shape
    s = torch.empty((m, n), dtype=x.dtype, device=x.device)
    b_t = b.transpose(-2, -1)

    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_M']), triton.cdiv(n, meta['BLOCK_N']))

    fused_diffusion[grid](b_t, x, mass, e, d, s,
                          m, n, k,
                          b_t.stride(0), b_t.stride(1), x.stride(0), x.stride(1),
                          s.stride(0), s.stride(1),
                          BLOCK_M=32, BLOCK_N=32, BLOCK_K=64,
                          GROUP_SIZE=8,
                          num_warps=4, num_stages=4)
    output = torch.matmul(b, s)
    ctx.save_for_backward(s, b, mass, e, d)
    return output, s
    
  @staticmethod
  def backward(ctx, dO, dS):
    '''
    Performs the backward pass for the input X and
    diffusion times D
    '''
    s, b, m, e, d = ctx.saved_tensors
    # contributions from both dO and dS
    # does this need a kernel too?
    s_grad = dS + torch.matmul(b.transpose(-2, -1), dO)
    dD, dX = None, None
    dD = -(e.unsqueeze(-1) * s_grad * s).sum(dim=0)
    if ctx.needs_input_grad[1]:
      # the input X doesn't always need grads, eg first layer
      # This could be a kernel too
      t = torch.exp(-e.unsqueeze(-1)*d)
      dst = s_grad*t
      dX = torch.matmul(b*m.unsqueeze(-1), dst)
    return None, dX, None, None, dD

# the function that callers can import and use
diffusion_func = FlashDiffusion.apply