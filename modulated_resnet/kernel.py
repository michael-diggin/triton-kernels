import torch
import triton
import triton.language as tl


@triton.jit
def _moderesnet_block_kernel(
    # pointers to inputs
    mod_act_ptr, syn_act_ptr, mod_w_ptr, syn_w_ptr,
    # pointers to output
    z1_ptr, z2_ptr,
    # Matrix dimensions
    M, N, K,
    # matrix strides
    stride_modact_m, stride_modact_k, stride_synact_m, stride_synact_k,
    stride_modw_n, stride_modw_k, stride_synw_n, stride_synw_k,
    stride_z1m, stride_z1n, stride_z2m, stride_z2n, 
    # block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    # meta params
    GROUP_SIZE: tl.constexpr,
):

    # ModAct is a [M, K] matrix (input)
    # SynAct is a [M, K] matrix (input)
    # ModW is a [N, K] matrix (input)
    # SynW is a [N, K] matrix (input)
    # Z1 is a [M, N] matrix (output)
    # Z2 is a [M, N] matrix (output)

    # Computes the following
    # tmp1 = SiLU(ModAct @ ModW.T)
    # tmp2 = SiLU(SynAct @ SynW.T)
    # tmp3 = tmp1*tmp2
    # Z1 = tmp1, Z2 = tmp3

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
    mod_act_ptrs = mod_act_ptr + offs_m[:, None]*stride_modact_m + offs_k_c[None, :]*stride_modact_k
    syn_act_ptrs = syn_act_ptr + offs_m[:, None]*stride_synact_m + offs_k_c[None, :]*stride_synact_k

    # ModW and SynW are accessed in a transposed manner
    modw_ptrs = mod_w_ptr + offs_k_c[None, :]*stride_modw_k + offs_n[:, None]*stride_modw_n
    synw_ptrs = syn_w_ptr + offs_k_c[None, :]*stride_synw_k + offs_n[:, None]*stride_synw_n

    act_mask = (z_row_indices < M)[:, None]
    w_mask = (z_col_indices < N)[:, None]

    # Initialize accumulators for the (BLOCK_M, BLOCK_N) block in the output
    # This is in float32 to avoid losing precision
    mod_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    syn_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Loop over the K dimension in chunks of BLOCK_K
    for k in range(0, tl.cdiv(K, BLOCK_K)):
      lim = K - k*BLOCK_K

      act_block = tl.load(mod_act_ptrs, mask=(offs_k[None, :] < lim) & act_mask, other=0.0)
      w_block = tl.load(modw_ptrs, mask=(offs_k[None, :] < lim) & w_mask, other=0.0).trans()
      mod_acc += tl.dot(act_block, w_block)

      act_block = tl.load(syn_act_ptrs, mask=(offs_k[None, :] < lim) & act_mask, other=0.0)
      w_block = tl.load(synw_ptrs, mask=(offs_k[None, :] < lim) & w_mask, other=0.0).trans()
      syn_acc += tl.dot(act_block, w_block)

      mod_act_ptrs += BLOCK_K*stride_modact_k
      modw_ptrs += BLOCK_K*stride_modw_k
      syn_act_ptrs += BLOCK_K*stride_synact_k
      synw_ptrs += BLOCK_K*stride_synw_k

    # second step - SiLU and pointwise multiplication
    mod_acc = mod_acc / (1.0 + tl.exp(-mod_acc))
    syn_acc = syn_acc / (1.0 + tl.exp(-syn_acc))
    syn_acc = syn_acc*mod_acc
    

    out_mask = (z_row_indices < M)[:, None] & (z_col_indices < N)[None, :]
    out_ptrs = z1_ptr + offs_m[:, None]*stride_z1m + offs_n[None, :]*stride_z1n
    tl.store(out_ptrs, mod_acc.to(z1_ptr.dtype.element_ty), mask=out_mask)

    out_ptrs = z2_ptr + offs_m[:, None]*stride_z2m + offs_n[None, :]*stride_z2n
    tl.store(out_ptrs, syn_acc.to(z2_ptr.dtype.element_ty), mask=out_mask)
