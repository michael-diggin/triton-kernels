import torch
import triton

from .torch import modresnet_fwd
from .kernel import triton_modresnet

torch.set_defaut_device("cuda")
torch.set_defaut_dtype(torch.bfloat16)

providers = ["Kernel", "Torch"]

configs = [triton.testing.Benchmark(
            x_names=["B"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 200, 5)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            line_vals=providers,
            line_names=providers,
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="bench",
            args={},
)]

@triton.testing.perf_report(configs)
def benchmark(B, provider):
    HD = 512
    mod_act = torch.rand(B, HD) / HD
    syn_act = torch.rand(B, HD) / HD
    mod_weight = torch.rand(HD, HD)
    syn_weight = torch.rand(HD, HD)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "Torch":
      ms, min_ms, max_ms = triton.testing.do_bench(lambda: modresnet_fwd(mod_act, syn_act, mod_weight, syn_weight), quantiles=quantiles)
    if provider == "Kernel":
      ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_modresnet(mod_act, syn_act, mod_weight, syn_weight), quantiles=quantiles)

    # leading TFLOPS term
    perf = lambda ms: (4 * B * HD * HD) * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=False)