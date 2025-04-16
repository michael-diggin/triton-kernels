import torch
from torch.amp import custom_fwd
from typing import Tuple
import triton
from .kernel import _moderesnet_block_kernel

class ModulatedResNetFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.bfloat16)
    def forward(ctx, mod_act: torch.Tensor, syn_act: torch.Tensor, mod_weight: torch.Tensor, syn_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get dimensions
        m, k = mod_act.shape
        n, _ = mod_weight.shape

        # Allocate output
        z1 = torch.empty_like(mod_act)
        z2 = torch.empty_like(syn_act)

        # Launch kernel
        grid = lambda meta: (
            triton.cdiv(m, meta['BLOCK_M']), 
            triton.cdiv(n, meta['BLOCK_N'])
        )
        
        _moderesnet_block_kernel[grid](
            mod_act, syn_act, mod_weight, syn_weight,
            z1, z2,
            m, n, k,
            mod_act.stride(0), mod_act.stride(1),
            syn_act.stride(0), syn_act.stride(1),
            mod_weight.stride(0), mod_weight.stride(1),
            syn_weight.stride(0), syn_weight.stride(1),
            z1.stride(0), z1.stride(1), z2.stride(0), z2.stride(1),
            GROUP_SIZE=8,
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=32,
        )
        return z1, z2

modulated_resnet_forward = ModulatedResNetFunction.apply
