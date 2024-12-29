import triton
import triton.language as tl

@triton.jit
def _gradient_matmul(
    gr_ptr, gi_ptr, x_ptr,
    zr_ptr, zi_ptr,
    M, N, K,
    stride_grm, stride_grk, stride_gim, stride_gik, stride_xk, stride_xn,
    stride_zrm, stride_zrn, stride_zim, stride_zin,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    
    # GR is a [M, K] matrix (input)
    # GI is a [M, K] matrix (input)
    # X is a [K, N] matrix (input) 
    # ZR is a [M, N] matrix (output)
    # ZI is a [M, N] matrix (output)

    # ZR = GR @ X
    # ZI = GI @ X

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)
    pid_0, pid_1 = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE)

    offs_k = tl.arange(0, BLOCK_K)

    z_row_indices = pid_0 * BLOCK_M + tl.arange(0, BLOCK_M)
    z_col_indices = pid_1 * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_m = tl.max_contiguous(tl.multiple_of(z_row_indices, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(z_col_indices, BLOCK_N), BLOCK_N)
    offs_k_c = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    gr_ptrs = gr_ptr + offs_m[:, None]*stride_grm + offs_k_c[None, :]*stride_grk
    gi_ptrs = gi_ptr + offs_m[:, None]*stride_gim + offs_k_c[None, :]*stride_gik
    x_ptrs = x_ptr + offs_k_c[:, None]*stride_xk + offs_n[None, :]*stride_xn

    g_mask = (z_row_indices < M)[:, None]
    x_mask = (z_col_indices < N)[None, :]

    acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    lim = K
    gr_k = BLOCK_K*stride_grk
    gi_k = BLOCK_K*stride_gik
    x_k = BLOCK_K*stride_xk

    # Loop over the K dimension in chunks of BLOCK_K
    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):

        gr_block = tl.load(gr_ptrs, mask=(offs_k[None, :] < lim) & g_mask, other=0.0)
        gi_block = tl.load(gi_ptrs, mask=(offs_k[None, :] < lim) & g_mask, other=0.0)
        x_block = tl.load(x_ptrs, mask=(offs_k[:, None] < lim) & x_mask, other=0.0)

        # Accumulate the result of the dot products
        acc_r += tl.dot(gr_block, x_block)
        acc_i += tl.dot(gi_block, x_block)

        gr_ptrs += gr_k
        gi_ptrs += gi_k
        x_ptrs += x_k
        lim -= BLOCK_K

    acc_r = acc_r.to(zr_ptr.dtype.element_ty)
    acc_i = acc_i.to(zi_ptr.dtype.element_ty)

    zr_ptrs = zr_ptr + offs_m[:, None]*stride_zrm + offs_n[None, :]*stride_zrn
    zi_ptrs = zi_ptr + offs_m[:, None]*stride_zim + offs_n[None, :]*stride_zin
    z_mask = (z_row_indices < M)[:, None] & (z_col_indices < N)[None, :]

    tl.store(zr_ptrs, acc_r, mask=z_mask)
    tl.store(zi_ptrs, acc_i, mask=z_mask)


@triton.jit
def _linear_matmul(
    xgr_ptr, xgi_ptr, lr_ptr, li_ptr,
    z_ptr,
    M, N, K,
    stride_xgrm, stride_xgrk, stride_xgim, stride_xgik, stride_lrn, stride_lrk, stride_lin, stride_lik,
    stride_zm, stride_zn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    
    # XGR is a [M, K] matrix (input)
    # XGI is a [M, K] matrix (input)
    # LR is a [N, K] matrix (input) 
    # LI is a [N, K] matrix (input) 
    # Z is a [M, N] matrix (output)

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)
    pid_0, pid_1 = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE)

    offs_k = tl.arange(0, BLOCK_K)

    z_row_indices = pid_0 * BLOCK_M + tl.arange(0, BLOCK_M)
    z_col_indices = pid_1 * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_m = tl.max_contiguous(tl.multiple_of(z_row_indices, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(z_col_indices, BLOCK_N), BLOCK_N)
    offs_k_c = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    xgr_ptrs = xgr_ptr + offs_m[:, None]*stride_xgrm + offs_k_c[None, :]*stride_xgrk
    xgi_ptrs = xgi_ptr + offs_m[:, None]*stride_xgim + offs_k_c[None, :]*stride_xgik

    # LR and LI are accessed in a transposed manner
    lr_ptrs = lr_ptr + offs_k_c[None, :]*stride_lrk + offs_n[:, None]*stride_lrn
    li_ptrs = li_ptr + offs_k_c[None, :]*stride_lik + offs_n[:, None]*stride_lin
    
    xg_mask = (z_row_indices < M)[:, None]
    l_mask = (z_col_indices < N)[:, None]

    acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    lim = K
    xgr_k = BLOCK_K*stride_xgrk
    xgi_k = BLOCK_K*stride_xgik
    lr_k = BLOCK_K*stride_lrk
    li_k = BLOCK_K*stride_lik

    # Loop over the K dimension in chunks of BLOCK_K
    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):

        xgr_block = tl.load(xgr_ptrs, mask=(offs_k[None, :] < lim) & xg_mask, other=0.0)
        xgi_block = tl.load(xgi_ptrs, mask=(offs_k[None, :] < lim) & xg_mask, other=0.0)
        lr_block = tl.load(lr_ptrs, mask=(offs_k[None, :] < lim) & l_mask, other=0.0).trans()
        li_block = tl.load(li_ptrs, mask=(offs_k[None, :] < lim) & l_mask, other=0.0).trans()

        # Accumulate the result of the dot products
        acc_r = acc_r + tl.dot(xgr_block, lr_block) - tl.dot(xgi_block, li_block)
        acc_i = acc_i + tl.dot(xgr_block, li_block) + tl.dot(xgi_block, lr_block)

        xgr_ptrs += xgr_k
        xgi_ptrs += xgi_k
        lr_ptrs += lr_k
        li_ptrs += li_k
        lim -= BLOCK_K

    # second step - reload xgr and xgi blocks
    xgr_ptrs = xgr_ptr + offs_m[:, None]*stride_xgrm + offs_n[None, :]*stride_xgrk
    xgi_ptrs = xgi_ptr + offs_m[:, None]*stride_xgim + offs_n[None, :]*stride_xgik
    xg_mask = (z_row_indices < M)[:, None] & (z_col_indices < N)[None, :]

    xgr_block = tl.load(xgr_ptrs, mask=xg_mask, other=0.0).to(tl.float32)
    xgi_block = tl.load(xgi_ptrs, mask=xg_mask, other=0.0).to(tl.float32)

    z = xgr_block*acc_r + xgi_block*acc_i
    z = tl.exp(2*z)
    z = (z-1)/(z+1)
    z = z.to(z_ptr.dtype.element_ty)

    out_ptrs = z_ptr + offs_m[:, None]*stride_zm + offs_n[None, :]*stride_zn

    tl.store(out_ptrs, z, mask=xg_mask)

#### Backward Pass Kernels

@triton.jit
def _bwd_kernel1(
    z_ptr, dz_ptr, xgr_ptr, xgi_ptr, lr_ptr, li_ptr,
    z_real_ptr, z_im_ptr, y_real_ptr, y_im_ptr,
    M, N, K,
    stride_zm, stride_zn, stride_dzm, stride_dzn, stride_xgrm, stride_xgrk, stride_xgim, stride_xgik, stride_lrn, stride_lrk, stride_lin, stride_lik,
    stride_zrm, stride_zrn, stride_zim, stride_zin, stride_yrm, stride_yrn, stride_yim, stride_yin,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    
    # Z is a [M, N] matrix (input)
    # DZ is a [M, N] matrix (input)
    # XGR is a [M, K] matrix (input)
    # XGI is a [M, K] matrix (input)
    # LR is a [N, K] matrix (input) 
    # LI is a [N, K] matrix (input) 
    # Z_REAL is a [M, N] matrix (output)
    # Z_IM is a [M, N] matrix (output)
    # Y_REAL is a [M, N] matrix (output)
    # Y_IM is a [M, N] matrix (output)
    
    # grad_x_dots = (1-z**2)*dZ
    # x_real = xgr@lr.T - xgi@li.T
    # x_im = xgr@li.T + xgi@lr.T
    # z_real = grad_x_dots * x_real
    # z_im = grad_x_dots * x_im
    # y_real = grad_x_dots * xgr
    # y_im = grad_x_dots * xgi


    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)
    pid_0, pid_1 = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE)

    offs_k = tl.arange(0, BLOCK_K)

    z_row_indices = pid_0 * BLOCK_M + tl.arange(0, BLOCK_M)
    z_col_indices = pid_1 * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_m = tl.max_contiguous(tl.multiple_of(z_row_indices, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(z_col_indices, BLOCK_N), BLOCK_N)
    offs_k_c = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    xgr_ptrs = xgr_ptr + offs_m[:, None]*stride_xgrm + offs_k_c[None, :]*stride_xgrk
    xgi_ptrs = xgi_ptr + offs_m[:, None]*stride_xgim + offs_k_c[None, :]*stride_xgik

    # LR and LI are accessed in a transposed manner
    lr_ptrs = lr_ptr + offs_k_c[None, :]*stride_lrk + offs_n[:, None]*stride_lrn
    li_ptrs = li_ptr + offs_k_c[None, :]*stride_lik + offs_n[:, None]*stride_lin
    
    xg_mask = (z_row_indices < M)[:, None]
    l_mask = (z_col_indices < N)[:, None]

    acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    lim = K
    xgr_k = BLOCK_K*stride_xgrk
    xgi_k = BLOCK_K*stride_xgik
    lr_k = BLOCK_K*stride_lrk
    li_k = BLOCK_K*stride_lik

    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):

        xgr_block = tl.load(xgr_ptrs, mask=(offs_k[None, :] < lim) & xg_mask, other=0.0)
        xgi_block = tl.load(xgi_ptrs, mask=(offs_k[None, :] < lim) & xg_mask, other=0.0)
        lr_block = tl.load(lr_ptrs, mask=(offs_k[None, :] < lim) & l_mask, other=0.0).trans()
        li_block = tl.load(li_ptrs, mask=(offs_k[None, :] < lim) & l_mask, other=0.0).trans()

        # Accumulate the result of the dot products
        acc_r = acc_r + tl.dot(xgr_block, lr_block) - tl.dot(xgi_block, li_block)
        acc_i = acc_i + tl.dot(xgr_block, li_block) + tl.dot(xgi_block, lr_block)

        xgr_ptrs += xgr_k
        xgi_ptrs += xgi_k
        lr_ptrs += lr_k
        li_ptrs += li_k
        lim -= BLOCK_K

    # second step - load grad block and compute z_real/z_im
    z_ptrs = z_ptr + offs_m[:, None]*stride_zm + offs_n[None, :]*stride_zn
    dz_ptrs = dz_ptr + offs_m[:, None]*stride_dzm + offs_n[None, :]*stride_dzn
    out_mask = xg_mask & l_mask
    z_block = tl.load(z_ptrs, mask=out_mask, other=0.0).to(tl.float32)
    grad_block = tl.load(dz_ptrs, mask=out_mask, other=0.0).to(tl.float32)

    grad_block = (1-z_block*z_block)*grad_block

    out_block = grad_block*acc_r
    out_ptrs = z_real_ptr + offs_m[:, None]*stride_zrm + offs_n[None, :]*stride_zrn
    tl.store(out_ptrs, out_block.to(z_real_ptr.dtype.element_ty), mask=out_mask)

    out_block = grad_block*acc_i
    out_ptrs = z_im_ptr + offs_m[:, None]*stride_zim + offs_n[None, :]*stride_zin
    tl.store(out_ptrs, out_block.to(z_im_ptr.dtype.element_ty), mask=out_mask)

    # third step - reload xgr and xgi blocks and compute y_real/y_im
    block_ptrs = xgr_ptr + offs_m[:, None]*stride_xgrm + offs_n[None, :]*stride_xgrk
    block = tl.load(block_ptrs, mask=out_mask, other=0.0).to(tl.float32)
    out_block = grad_block*block
    out_ptrs = y_real_ptr + offs_m[:, None]*stride_yrm + offs_n[None, :]*stride_yrn
    tl.store(out_ptrs, out_block.to(y_real_ptr.dtype.element_ty), mask=out_mask)

    block_ptrs = xgi_ptr + offs_m[:, None]*stride_xgim + offs_n[None, :]*stride_xgik
    block = tl.load(block_ptrs, mask=out_mask, other=0.0).to(tl.float32)
    out_block = grad_block*block
    out_ptrs = y_im_ptr + offs_m[:, None]*stride_yim + offs_n[None, :]*stride_yin
    tl.store(out_ptrs, out_block.to(y_im_ptr.dtype.element_ty), mask=out_mask)

@triton.jit
def _bwd_kernel2(
    xgr_ptr, xgi_ptr, grad_real_ptr, grad_im_ptr,
    dlr_ptr, dli_ptr,
    M, N, K,
    stride_xgrk, stride_xgrm, stride_xgik, stride_xgim, stride_gradrk, stride_gradrn, stride_gradik, stride_gradin,
    stride_dlrm, stride_dlrn, stride_dlim, stride_dlin,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    
    # XGR is a [K, M] matrix (input)
    # XGI is a [K, M] matrix (input)
    # GRAD_REAL is a [K, N] matrix (input) 
    # GRAD_IM is a [K, N] matrix (input) 
    # DLR is a [M, N] matrix (output)
    # DLI is a [M, N] matrix (output)

    # DLR = XGR.T @ GRAD_REAL + XGI.T @ GRAD_IM
    # DLI = XGR.T @ GRAD_IM - XGI.T @ GRAD_REAL


    # while the lhs of some matmuls are transposed
    # swizzling doesn't seem to help
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)

    z_row_indices = pid_0 * BLOCK_M + tl.arange(0, BLOCK_M)
    z_col_indices = pid_1 * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_m = tl.max_contiguous(tl.multiple_of(z_row_indices, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(z_col_indices, BLOCK_N), BLOCK_N)
    offs_k_c = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    xgr_ptrs = xgr_ptr + offs_m[None, :]*stride_xgrm + offs_k_c[:, None]*stride_xgrk
    xgi_ptrs = xgi_ptr + offs_m[None, :]*stride_xgim + offs_k_c[:, None]*stride_xgik
    grad_real_ptrs = grad_real_ptr + offs_k_c[:, None]*stride_gradrk + offs_n[None, :]*stride_gradrn
    grad_im_ptrs = grad_im_ptr + offs_k_c[:, None]*stride_gradik + offs_n[None, :]*stride_gradin

    xg_mask = (z_row_indices < M)[None, :]
    grad_mask = (z_col_indices < N)[None, :]

    acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    lim = K
    xgr_k = BLOCK_K*stride_xgrk
    xgi_k = BLOCK_K*stride_xgik
    grad_real_k = BLOCK_K*stride_gradrk
    grad_im_k = BLOCK_K*stride_gradik

    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):

        xgr_block = tl.load(xgr_ptrs, mask=(offs_k[:, None] < lim) & xg_mask, other=0.0).trans()
        xgi_block = tl.load(xgi_ptrs, mask=(offs_k[:, None] < lim) & xg_mask, other=0.0).trans()
        grad_real_block = tl.load(grad_real_ptrs, mask=(offs_k[:, None] < lim) & grad_mask, other=0.0)
        grad_im_block = tl.load(grad_im_ptrs, mask=(offs_k[:, None] < lim) & grad_mask, other=0.0)

        acc_r = acc_r + tl.dot(xgr_block, grad_real_block) + tl.dot(xgi_block, grad_im_block)
        acc_i = acc_i + tl.dot(xgr_block, grad_im_block) - tl.dot(xgi_block, grad_real_block)

        xgr_ptrs += xgr_k
        xgi_ptrs += xgi_k
        grad_real_ptrs += grad_real_k
        grad_im_ptrs += grad_im_k
        lim -= BLOCK_K

    acc_r = acc_r.to(dlr_ptr.dtype.element_ty)
    acc_i = acc_i.to(dli_ptr.dtype.element_ty)

    dlr_ptrs = dlr_ptr + offs_m[:, None]*stride_dlrm + offs_n[None, :]*stride_dlrn
    dli_ptrs = dli_ptr + offs_m[:, None]*stride_dlim + offs_n[None, :]*stride_dlin
    z_mask = (z_row_indices < M)[:, None] & (z_col_indices < N)[None, :]

    tl.store(dlr_ptrs, acc_r, mask=z_mask)
    tl.store(dli_ptrs, acc_i, mask=z_mask)

@triton.jit
def _bwd_kernel3(
    gr_ptr, gi_ptr, lr_ptr, li_ptr, grads_real_ptr, grads_im_ptr,
    zr_ptr, zi_ptr,
    M, N, K,
    stride_grm, stride_grk, stride_gim, stride_gik, stride_lrk, stride_lrn, stride_lik, stride_lin,
    stride_gradsrm, stride_gradsrn, stride_gradsim, stride_gradsin,
    stride_zrm, stride_zrn, stride_zim, stride_zin,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    
    # GR is a [M, K] matrix (input)
    # GI is a [M, K] matrix (input)
    # X is a [K, N] matrix (input) 
    # ZR is a [M, N] matrix (output)
    # ZI is a [M, N] matrix (output)

    # zr = gr @ lr + gi @ li + grads_real
    # zi = gi @ lr - gr @ li + grads_im


    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)
    pid_0, pid_1 = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE)

    offs_k = tl.arange(0, BLOCK_K)

    z_row_indices = pid_0 * BLOCK_M + tl.arange(0, BLOCK_M)
    z_col_indices = pid_1 * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_m = tl.max_contiguous(tl.multiple_of(z_row_indices, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(z_col_indices, BLOCK_N), BLOCK_N)
    offs_k_c = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    gr_ptrs = gr_ptr + offs_m[:, None]*stride_grm + offs_k_c[None, :]*stride_grk
    gi_ptrs = gi_ptr + offs_m[:, None]*stride_gim + offs_k_c[None, :]*stride_gik
    lr_ptrs = lr_ptr + offs_k_c[:, None]*stride_lrk + offs_n[None, :]*stride_lrn
    li_ptrs = li_ptr + offs_k_c[:, None]*stride_lik + offs_n[None, :]*stride_lin

    g_mask = (z_row_indices < M)[:, None]
    l_mask = (z_col_indices < N)[None, :]

    acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    lim = K
    gr_k = BLOCK_K*stride_grk
    gi_k = BLOCK_K*stride_gik
    lr_k = BLOCK_K*stride_lrk
    li_k = BLOCK_K*stride_lik

    # Loop over the K dimension in chunks of BLOCK_K
    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):

        gr_block = tl.load(gr_ptrs, mask=(offs_k[None, :] < lim) & g_mask, other=0.0)
        gi_block = tl.load(gi_ptrs, mask=(offs_k[None, :] < lim) & g_mask, other=0.0)
        lr_block = tl.load(lr_ptrs, mask=(offs_k[:, None] < lim) & l_mask, other=0.0)
        li_block = tl.load(li_ptrs, mask=(offs_k[:, None] < lim) & l_mask, other=0.0)

        # Accumulate the result of the dot products
        acc_r = acc_r + tl.dot(gr_block, lr_block) + tl.dot(gi_block, li_block)
        acc_i = acc_i + tl.dot(gi_block, lr_block) - tl.dot(gr_block, li_block)

        gr_ptrs += gr_k
        gi_ptrs += gi_k
        lr_ptrs += lr_k
        li_ptrs += li_k
        lim -= BLOCK_K

    out_mask = g_mask & l_mask

    grads_r_ptrs = grads_real_ptr + offs_m[:, None]*stride_gradsrm + offs_n[None, :]*stride_gradsrn
    grads_i_ptrs = grads_im_ptr + offs_m[:, None]*stride_gradsim + offs_n[None, :]*stride_gradsin
    
    grads_r_block = tl.load(grads_r_ptrs, mask=out_mask, other=0.0)
    grads_i_block = tl.load(grads_i_ptrs, mask=out_mask, other=0.0)

    acc_r = acc_r + grads_r_block
    acc_i = acc_i + grads_i_block

    zr_ptrs = zr_ptr + offs_m[:, None]*stride_zrm + offs_n[None, :]*stride_zrn
    zi_ptrs = zi_ptr + offs_m[:, None]*stride_zim + offs_n[None, :]*stride_zin

    tl.store(zr_ptrs, acc_r.to(zr_ptr.dtype.element_ty), mask=out_mask)
    tl.store(zi_ptrs, acc_i.to(zi_ptr.dtype.element_ty), mask=out_mask)

@triton.jit
def _bwd_kernel4(
    gr_ptr, gi_ptr, grad_real_ptr, grad_im_ptr,
    dx_ptr,
    M, N, K,
    stride_grk, stride_grm, stride_gik, stride_gim, stride_gradrk, stride_gradrn, stride_gradik, stride_gradin,
    stride_dxm, stride_dxn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    
    # GR is a [K, M] matrix (input)
    # GI is a [K, M] matrix (input)
    # GRAD_REAL is a [K, N] matrix (input) 
    # GRAD_IM is a [K, N] matrix (input) 
    # DX is a [M, N] matrix (output)

    # dX = gr.T @ grad_real + gi.T @ grad_im)

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)
    pid_0, pid_1 = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE)

    offs_k = tl.arange(0, BLOCK_K)

    z_row_indices = pid_0 * BLOCK_M + tl.arange(0, BLOCK_M)
    z_col_indices = pid_1 * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_m = tl.max_contiguous(tl.multiple_of(z_row_indices, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(z_col_indices, BLOCK_N), BLOCK_N)
    offs_k_c = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    gr_ptrs = gr_ptr + offs_m[None, :]*stride_grm + offs_k_c[:, None]*stride_grk
    gi_ptrs = gi_ptr + offs_m[None, :]*stride_gim + offs_k_c[:, None]*stride_gik
    grad_real_ptrs = grad_real_ptr + offs_k_c[:, None]*stride_gradrk + offs_n[None, :]*stride_gradrn
    grad_im_ptrs = grad_im_ptr + offs_k_c[:, None]*stride_gradik + offs_n[None, :]*stride_gradin

    g_mask = (z_row_indices < M)[None, :]
    grad_mask = (z_col_indices < N)[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    lim = K
    gr_k = BLOCK_K*stride_grk
    gi_k = BLOCK_K*stride_gik
    grad_real_k = BLOCK_K*stride_gradrk
    grad_im_k = BLOCK_K*stride_gradik

    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):

        gr_block = tl.load(gr_ptrs, mask=(offs_k[:, None] < lim) & g_mask, other=0.0).trans()
        gi_block = tl.load(gi_ptrs, mask=(offs_k[:, None] < lim) & g_mask, other=0.0).trans()
        grad_real_block = tl.load(grad_real_ptrs, mask=(offs_k[:, None] < lim) & grad_mask, other=0.0)
        grad_im_block = tl.load(grad_im_ptrs, mask=(offs_k[:, None] < lim) & grad_mask, other=0.0)

        # Accumulate the result of the dot products
        acc = acc + tl.dot(gr_block, grad_real_block) + tl.dot(gi_block, grad_im_block)

        gr_ptrs += gr_k
        gi_ptrs += gi_k
        grad_real_ptrs += grad_real_k
        grad_im_ptrs += grad_im_k
        lim -= BLOCK_K

    acc = acc.to(dx_ptr.dtype.element_ty)

    dx_ptrs = dx_ptr + offs_m[:, None]*stride_dxm + offs_n[None, :]*stride_dxn
    z_mask = g_mask & grad_mask

    tl.store(dx_ptrs, acc, mask=z_mask)