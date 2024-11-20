
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config


torch._functorch.config.debug_partitioner = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None



# torch version: 2.5.1+cu121
# torch cuda version: 12.1
# torch git version: a8d6afb511a69687bbb2b7e88a3cf67917e1697e


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Tue_Aug_15_22:02:13_PDT_2023 
# Cuda compilation tools, release 12.2, V12.2.140 
# Build cuda_12.2.r12.2/compiler.33191640_0 

# GPU Hardware Info: 
# NVIDIA A100-SXM4-40GB : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
        permute = torch.ops.aten.permute.default(arg0_1, [1, 0])
        unsqueeze = torch.ops.aten.unsqueeze.default(arg1_1, -1);  arg1_1 = None
        mul = torch.ops.aten.mul.Tensor(arg2_1, unsqueeze);  arg2_1 = unsqueeze = None
        mm = torch.ops.aten.mm.default(permute, mul);  permute = mul = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        neg = torch.ops.aten.neg.default(unsqueeze_1);  unsqueeze_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(neg, arg4_1);  neg = arg4_1 = None
        exp = torch.ops.aten.exp.default(mul_1);  mul_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(exp, mm);  exp = mm = None
        mm_1 = torch.ops.aten.mm.default(arg0_1, mul_2);  arg0_1 = None
        return (mm_1, mul_2)
        
def load_args(reader):
    buf0 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (2048, 128), dtype=torch.bfloat16, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (2048,), dtype=torch.bfloat16, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 1048576, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (2048, 256), dtype=torch.bfloat16, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (128,), dtype=torch.bfloat16, is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 512, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (256,), dtype=torch.bfloat16, is_leaf=True)  # arg4_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)