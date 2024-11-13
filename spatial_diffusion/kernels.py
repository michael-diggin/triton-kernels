import triton
import triton.language as tl


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
    # This Kernel computes a large amount of the Spatial Diffusion Layer.
    # Specifically it fuses the `in_basis` part, with the diffusion part, to compute
    # the `Spectral` or `in_basis` output tensor.

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
    offs_k = tl.arange(0, BLOCK_K)

    # Compute row and column indices for the output Z block
    z_row_indices = pid_0 * BLOCK_M + tl.arange(0, BLOCK_M)
    z_col_indices = pid_1 * BLOCK_N + tl.arange(0, BLOCK_N)

    # hint to the compiler that these are contiguous
    offs_m = tl.max_contiguous(tl.multiple_of(z_row_indices, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(z_col_indices, BLOCK_N), BLOCK_N)
    offs_k_c = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    # set up the pointers to the first blocks that get loaded
    # the stride values are in case the tensors aren't contiguous in memory
    # eg. the Basis vector is given as a transposed matrix so care is needed
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

    # cast back to the dtype of the input
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


@triton.jit
def dd_ds_bwd(
    # pointers to inputs
    ds_ptr, do_ptr, b_ptr, s_ptr, e_ptr, d_ptr,
    # pointers to outputs
    y_ptr, z_ptr,
    # Matrix dimensions
    M, N, K,
    # matrix strides
    stride_dsm, stride_dsn, stride_dok, stride_don, stride_bk, stride_bm, stride_sm, stride_sn,
    stride_zm, stride_zn,
    # block sizes -> typically 128, 32, 32
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    # meta params
    GROUP_SIZE: tl.constexpr,
):
    
    # dS is a [M, N] matrix (input)
    # dO is a [V, N] matrix (input)
    # B is a [M, K] matrix (input)
    # S is a [M, N] matrix (input)
    # E is a [M] row vector (input)
    # D is a [N] row vector (input)
    # Y is a [N] row vector (output)
    # Z is a [M, N] matrix (output)

    # X = ds + B@dO
    # Y = (-E.unsqueeze(-1)*S*X).sum(dim=0)
    # Z = X*exp(-E.unsqueeze(-1)*d)

    # Since know that M is 128 and hence small enough to fit in SRAM
    # we use BLOCK_M=128 as well and perform the Y sum within the block

    # Map program ids `pid` to the block of Z it should compute.
    # This is done in a swizzled/grouped ordering to promote L2 data reuse.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)
    pid_0, pid_1 = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE)

    # Offsets for the K dimension (looped dimension)
    offs_k = tl.arange(0, BLOCK_K)  # range within the K block

    # Compute row and column indices for the output X block
    x_row_indices = pid_0 * BLOCK_M + tl.arange(0, BLOCK_M)
    x_col_indices = pid_1 * BLOCK_N + tl.arange(0, BLOCK_N)

    # hint to the compiler that these are contiguous
    offs_m = tl.max_contiguous(tl.multiple_of(x_row_indices, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(x_col_indices, BLOCK_N), BLOCK_N)
    offs_k_c = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    # set up the pointers to the first blocks that get loaded
    # the stride values are in case the tensors aren't contiguous in memory
    b_ptrs = b_ptr + offs_m[:, None]*stride_bm + offs_k_c[None, :]*stride_bk
    do_ptrs = do_ptr + offs_k_c[:, None]*stride_dok + offs_n[None, :]*stride_don

    b_mask = (x_row_indices < M)[:, None]
    do_mask = (x_col_indices < N)[None, :]

    # Initialize accumulator for this (BLOCK_M, BLOCK_N) block in the output
    # This is in float32 to avoid losing precision
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    lim = K
    b_k = BLOCK_K*stride_bk
    do_k = BLOCK_K*stride_dok

    # Loop over the K dimension in chunks of BLOCK_K
    for _ in range(0, tl.cdiv(K, BLOCK_K)):

      b_block = tl.load(b_ptrs, mask=(offs_k[None, :] < lim) & b_mask, other=0.0)
      do_block = tl.load(do_ptrs, mask=(offs_k[:, None] < lim) & do_mask, other=0.0)

      # Accumulate the result of the dot product
      accumulator += tl.dot(b_block, do_block)

      b_ptrs += b_k
      do_ptrs += do_k
      lim -= BLOCK_K

    accumulator.to(ds_ptr.dtype.element_ty)

    output_mask = (x_row_indices < M)[:, None] & (x_col_indices < N)[None, :]

    # second step - load dS block to add to accumulator
    ds_ptrs = ds_ptr + offs_m[:, None]*stride_dsm + offs_n[None, :]*stride_dsn
    ds_block = tl.load(ds_ptrs, mask=output_mask, other=0.0)

    accumulator += ds_block

    # third step - load the S and E blocks
    s_ptrs = s_ptr + offs_m[:, None]*stride_sm + offs_n[None, :]*stride_sn
    s_block = tl.load(s_ptrs, mask=output_mask, other=0.0)
    e_block = tl.load(e_ptr+offs_m, mask=x_row_indices<M, other=0.0).expand_dims(-1)

    # reduce along the 0 dimension and write to the Y row vector
    y_block = -1*e_block*s_block*accumulator
    y_sum = tl.sum(y_block, axis=0)
    tl.store(y_ptr + offs_n, y_sum, mask=(x_col_indices<N))

    # forth step - load D block to compute the exponentiated term
    d_block = tl.load(d_ptr + offs_n, mask=x_col_indices<N, other=0.0).expand_dims(0)

    # make use of the following identity - supposedly 'more stable'
    # tl.exp(x) = tl.exp2(log2(e)*x)
    minus_log2_e = -1.4426950408889634
    diff_coefs = tl.math.exp2(minus_log2_e*e_block*d_block)
    z_block = diff_coefs*accumulator

    # Set up pointers and masks for storing the second output block in Z
    z_ptrs = z_ptr + offs_m[:, None]*stride_zm + offs_n[None, :]*stride_zn
    # Store the accumulated result in Z
    tl.store(z_ptrs, z_block, mask=output_mask)


@triton.jit
def dx_bwd(
    # pointers to inputs
    b_ptr, s_ptr, mass_ptr,
    # pointers to outputs
    dx_ptr,
    # Matrix dimensions
    M, N, K,
    # matrix strides
    stride_bm, stride_bk, stride_sk, stride_sn,
    stride_dxm, stride_dxn,
    # block sizes -> probs 32, 32, 128
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    # meta params
    GROUP_SIZE: tl.constexpr,
):

    # B is a [M, K] matrix (input)
    # S is a [K, N] matrix (input)
    # MASS is a [M] row vector (input)
    # DX is a [M, N] matrix (output)

    # DX = MASS.unsqueeze(-1)*(B@dO)

    # Map program ids `pid` to the block of DX it should compute.
    # This is done in a swizzled/grouped ordering to promote L2 data reuse.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)
    pid_0, pid_1 = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE)

    # Offsets for the K dimension (looped dimension)
    offs_k = tl.arange(0, BLOCK_K)  # range within the K block

    # Compute row and column indices for the output DX block
    dx_row_indices = pid_0 * BLOCK_M + tl.arange(0, BLOCK_M)
    dx_col_indices = pid_1 * BLOCK_N + tl.arange(0, BLOCK_N)

    # hint to the compiler that these are contiguous
    offs_m = tl.max_contiguous(tl.multiple_of(dx_row_indices, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(dx_col_indices, BLOCK_N), BLOCK_N)
    offs_k_c = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    # set up the pointers to the first blocks that get loaded
    # the stride values are in case the tensors aren't contiguous in memory
    b_ptrs = b_ptr + offs_m[:, None]*stride_bm + offs_k_c[None, :]*stride_bk
    s_ptrs = s_ptr + offs_k_c[:, None]*stride_sk + offs_n[None, :]*stride_sn

    b_mask = (dx_row_indices < M)[:, None]
    s_mask = (dx_col_indices < N)[None, :]

    # Initialize accumulator for this (BLOCK_M, BLOCK_N) block in the output
    # This is in float32 to avoid losing precision
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    lim = K
    b_k = BLOCK_K*stride_bk
    s_k = BLOCK_K*stride_sk

    # Loop over the K dimension in chunks of BLOCK_K
    for _ in range(0, tl.cdiv(K, BLOCK_K)):

      b_block = tl.load(b_ptrs, mask=(offs_k[None, :] < lim) & b_mask, other=0.0)
      s_block = tl.load(s_ptrs, mask=(offs_k[:, None] < lim) & s_mask, other=0.0)

      # Accumulate the result of the dot product
      accumulator += tl.dot(b_block, s_block)

      b_ptrs += b_k
      s_ptrs += s_k
      lim -= BLOCK_K

    accumulator.to(s_ptr.dtype.element_ty)

    # load in the MASS block now
    mass_block = tl.load(mass_ptr + offs_m, mask=offs_m<M, other=0.0)
    accumulator = mass_block[:, None]*accumulator

    output_mask = (dx_row_indices < M)[:, None] & (dx_col_indices < N)[None, :]
    dx_ptrs = dx_ptr + offs_m[:, None]*stride_dxm + offs_n[None, :]*stride_dxn
    tl.store(dx_ptrs, accumulator, mask=output_mask)
