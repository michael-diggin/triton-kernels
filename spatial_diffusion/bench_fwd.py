import torch
import triton

from .torch import torch_diffusion
from .function import diffusion_func

if __name__ == '__main__':

    compiled = torch.compile(torch_diffusion)

    DTYPE = torch.bfloat16
    providers = ["Kernel_Func", "Compiled torch", "Torch"]

    configs = [triton.testing.Benchmark(
        x_names=["V"],  # Arg for number of verts
        x_vals=[128 * i for i in range(2, 33)],  # Different possible values V
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        line_vals=providers,
        line_names=providers,
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],
        ylabel="TFLOPS",
        plot_name="bench-fwd",
        args={"data_type": DTYPE}, # any extra args
    )]

    @triton.testing.perf_report(configs)
    def benchmark(V, provider, data_type):
        N = 256
        M = 128
        b = torch.randn((V, M), device='cuda', dtype=data_type) / V
        x = torch.randn((V, N), device='cuda', dtype=data_type) / V
        m = torch.randn((V), device='cuda', dtype=data_type)
        e = torch.randn((M), device='cuda', dtype=data_type)
        d = torch.randn((N), device='cuda', dtype=data_type)
        quantiles = [0.5, 0.2, 0.8]
        if provider == "Torch":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_diffusion(b, x, m, e, d), quantiles=quantiles)
        if provider == "Compiled torch":
          ms, min_ms, max_ms = triton.testing.do_bench(lambda: compiled(b, x, m, e, d), quantiles=quantiles)
        if provider == "Kernel_Func":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: diffusion_func(b, x, m, e, d), quantiles=quantiles)

        # matmul is 2*m.n.v FLOPS, and there are 2 of them
        # mat*vec elementwise is v.n
        # vec*vec is m.n
        # mat*mat is m.n
        perf = lambda ms: (4*M*N*V + 2*M*N + V*N) * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)


    benchmark.run(show_plots=True, print_data=True)

    '''
    Best config seems to be (at least for larger V)
    BLOCK_M: 32, BLOCK_N: 32, BLOCK_K: 64, num_warps: 4, num_ctas: 1, num_stages: 4, maxnreg: None
    '''