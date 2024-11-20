# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, grid_combo_kernels, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_root/p3/cp3rl3snkcgofapcsbsb4mii35xpq5j7a7gj3n3qngderytoil5v.py
# Topologically Sorted Source Nodes: [x_m], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   x_m => mul
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, %unsqueeze), kwargs = {})
triton_poi_fused_mul_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '54E46422D5DB2E55B804C8E038A4A0E2ECEED6FCC5402DED453936C14F5DFA13', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/wu/cwuasj7g3h76pxxotxu34unkz4on33htu7en3pwi6ljv6memayf6.py
# Topologically Sorted Source Nodes: [neg, mul_1, diffs, spectral], Original ATen: [aten.neg, aten.mul, aten.exp]
# Source node to ATen node mapping:
#   diffs => exp
#   mul_1 => mul_1
#   neg => neg
#   spectral => mul_2
# Graph fragment:
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%unsqueeze_1,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %arg4_1), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_1,), kwargs = {})
#   %mul_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp, %mm), kwargs = {})
triton_poi_fused_exp_mul_neg_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_exp_mul_neg_1', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '54E46422D5DB2E55B804C8E038A4A0E2ECEED6FCC5402DED453936C14F5DFA13', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 256)
    x0 = xindex % 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = -tmp0
    tmp3 = tmp1 * tmp2
    tmp4 = tl_math.exp(tmp3)
    tmp6 = tmp4 * tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2048, 128), (128, 1))
    assert_size_stride(arg1_1, (2048, ), (1, ))
    assert_size_stride(arg2_1, (2048, 256), (256, 1))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (256, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2048, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_m], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(arg2_1, arg1_1, buf0, 524288, grid=grid(524288), stream=stream0)
        del arg1_1
        del arg2_1
        buf1 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_m, in_basis], Original ATen: [aten.mul, aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg0_1, (128, 2048), (1, 128), 0), buf0, out=buf1)
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [neg, mul_1, diffs, spectral], Original ATen: [aten.neg, aten.mul, aten.exp]
        triton_poi_fused_exp_mul_neg_1.run(buf2, arg3_1, arg4_1, 32768, grid=grid(32768), stream=stream0)
        del arg3_1
        del arg4_1
        buf3 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.mm]
        extern_kernels.mm(arg0_1, buf2, out=buf3)
        del arg0_1
    return (buf3, buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2048, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
