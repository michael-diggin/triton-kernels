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
        plot_name="bench-bwd",
        args={"data_type": DTYPE},
    )]

    @triton.testing.perf_report(configs)
    def benchmark_bwd(V, provider, data_type):
        M = 128
        N = 256
        b = torch.randn((V, M), device='cuda', dtype=data_type) / V
        x = torch.randn((V, N), device='cuda', dtype=data_type) / V
        x.requires_grad_(True)
        m = torch.randn((V), device='cuda', dtype=data_type)
        e = torch.randn((M), device='cuda', dtype=data_type)
        d = torch.randn((N), device='cuda', dtype=data_type, requires_grad=True)
        quantiles = [0.5, 0.2, 0.8]

        dO = 0.1*torch.ones((V, N), device="cuda", dtype=data_type)
        dS = 0.1*torch.ones((M, N), device="cuda", dtype=data_type)

        if provider == "Torch":
          o, s = torch_diffusion(b, x, m, e, d)
          ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.autograd.backward([o, s],[dO, dS], retain_graph=True),
                                                       grad_to_none=[x, d], quantiles=quantiles, rep=500)
        if provider == "Compiled torch":
          o, s = compiled(b, x, m, e, d)
          ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.autograd.backward([o, s],[dO, dS], retain_graph=True),
                                                       grad_to_none=[x, d], quantiles=quantiles, rep=500)
        if provider == "Kernel":
          o, s = diffusion_func(b, x, m, e, d)
          ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.autograd.backward([o, s],[dO, dS], retain_graph=True),
                                                       grad_to_none=[x, d], quantiles=quantiles, rep=500)

        # Two matmuls (dS = b.T@dO and to get dX)
        # matrix*vec of V*M
        # 4 matrix*matrix or matrix*vec of M*N
        perf = lambda ms: (4*V*M*N + V*M + 4*M*N) * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)


    benchmark_bwd.run(show_plots=True, print_data=True)

