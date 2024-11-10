import torch

from .torch import torch_diffusion
from .kernel import diffusion

torch.manual_seed(0)
dtype = torch.bfloat16

k = 2048
m = 128
n = 256

X = torch.rand((k,m), dtype=dtype, device="cuda") / k # this is needed otherwise it would overflow bfloat16 and lead to rounding errors
Y = torch.rand((k,n), dtype=dtype, device="cuda") / k
E = torch.rand(m, dtype=dtype, device="cuda")
D = torch.rand(n, dtype=dtype, device="cuda")
MASS = torch.rand(k, dtype=dtype, device="cuda")

if __name__ == '__main__':

    expected, exp_spect = torch_diffusion(X, Y, MASS, E, D)
    Z, spect = diffusion(X, Y, MASS, E, D)

    atol = 1e-7 if dtype == torch.float32 else 1e-3
    rtol = 1e-5 if dtype == torch.float32 else 1e-2

    print(torch.allclose(expected, Z, atol=atol, rtol=rtol))
    print(torch.max(torch.abs(expected - Z)))
    print("\n")
    print(torch.allclose(exp_spect, spect, atol=atol, rtol=rtol))
    print(torch.max(torch.abs(exp_spect - spect)))