import torch
import triton

from .torch import torch_diffusion
from .kernel import diffusion

if __name__ == '__main__':

    compiled = torch.compile(torch_diffusion)

    DTYPE = torch.bfloat16

    configs = [triton.testing.Benchmark(
                x_names=["V"],  # Argument names to use as an x-axis for the plot
                x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
                line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
                # Possible values for `line_arg`
                line_vals=["Kernel", "Compiled torch", "Torch"],
                line_names=["Kernel", "Compiled torch", "Torch"],
                styles=[("green", "-"), ("blue", "-"), ("red", "-")],
                ylabel="TFLOPS",  # Label name for the y-axis
                plot_name="bench",
                args={"data_type": DTYPE},
    )]

    @triton.testing.perf_report(configs)
    def benchmark(V, provider, data_type):
        b = torch.randn((V, 128), device='cuda', dtype=data_type) / V
        x = torch.randn((V, 256), device='cuda', dtype=data_type) / V
        m = torch.randn((V), device='cuda', dtype=data_type)
        e = torch.randn((128), device='cuda', dtype=data_type)
        d = torch.randn((256), device='cuda', dtype=data_type)
        quantiles = [0.5, 0.2, 0.8]
        if provider == "Torch":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_diffusion(b, x, m, e, d), quantiles=quantiles)
        if provider == "Compiled torch":
          ms, min_ms, max_ms = triton.testing.do_bench(lambda: compiled(b, x, m, e, d), quantiles=quantiles)
        if provider == "Kernel":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: diffusion(b, x, m, e, d), quantiles=quantiles)

        # matmul is m.n.(2v-1) FLOPS
        # mat*vec elementwise is v.n
        # vec*vec is m.n
        # mat*mat is m.n
        N = 128
        M = 256
        perf = lambda ms: (2*M * N * (2*V -1) + 2*M*N + V*N) * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)


    benchmark.run(show_plots=True, print_data=True)

    '''
    Best config seems to be (at least for larger V)
    BLOCK_M: 32, BLOCK_N: 32, BLOCK_K: 64, num_warps: 4, num_ctas: 1, num_stages: 4, maxnreg: None
    '''