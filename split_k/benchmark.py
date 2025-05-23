import torch
import triton
from .split_k import splitk_kernel
from attempt1.kernel import first_kernel
from attempt2.kernel import second_kernel
from attempt3.kernel import third_kernel

DTYPE = torch.float16

providers = ["Torch", "First Kernel", "Second Kernel", "Third Kernel", "Final Kernel"]

configs = [triton.testing.Benchmark(
            x_names=["V"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 200, 5)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            line_vals=providers,
            line_names=providers,
            styles=[("green", "-"), ("cyan", "-"), ("red", "-"), ("orange", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="bench",
            args={"dtype": DTYPE},
)]

@triton.testing.perf_report(configs)
def benchmark(V, provider, dtype):
    W = 256
    M = 256
    X = torch.rand((M, V), device="cuda", dtype=dtype)
    Y = torch.rand((V, W), device="cuda", dtype=dtype)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "Torch":
      ms, min_ms, max_ms = triton.testing.do_bench(lambda: X@Y, quantiles=quantiles)
    if provider == "Final Kernel":
      ms, min_ms, max_ms = triton.testing.do_bench(lambda: splitk_kernel(X, Y), quantiles=quantiles)
    if provider == "First Kernel":
      ms, min_ms, max_ms = triton.testing.do_bench(lambda: first_kernel(X, Y), quantiles=quantiles)
    if provider == "Second Kernel":
      ms, min_ms, max_ms = triton.testing.do_bench(lambda: second_kernel(X, Y), quantiles=quantiles)
    if provider == "Third Kernel":
      ms, min_ms, max_ms = triton.testing.do_bench(lambda: third_kernel(X, Y), quantiles=quantiles)

    perf = lambda ms: (2 * M * W * V) * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=False)