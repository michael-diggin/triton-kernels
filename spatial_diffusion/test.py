# Script to verify that the new code produces the correct results
# up to minor precison differences
import torch

from .torch import torch_diffusion
from .kernel import diffusion_func

torch.manual_seed(0)
dtype = torch.bfloat16

k = 2048
m = 128
n = 256

X = torch.rand((k,m), dtype=dtype, device="cuda") / k # this is needed otherwise it would overflow bfloat16 and lead to rounding errors
BASE_INPUT = torch.rand((k,n), dtype=dtype, device="cuda") / k
Y_1 = BASE_INPUT.clone()
Y_1.requires_grad = True
Y_2 = BASE_INPUT.clone()
Y_2.requires_grad = True
MASS = torch.rand(k, dtype=dtype, device="cuda")
E = torch.rand(m, dtype=dtype, device="cuda")
BASE_D = torch.rand(n, dtype=dtype, device="cuda")
D_1 = BASE_D.clone()
D_1.requires_grad = True
D_2 = BASE_D.clone()
D_2.requires_grad = True

if __name__ == '__main__':

    expected, exp_spect = torch_diffusion(X, Y_1, MASS, E, D_1)
    Z, spect = diffusion_func(X, Y_2, MASS, E, D_2)

    atol = 1e-7 if dtype == torch.float32 else 1e-3
    rtol = 1e-5 if dtype == torch.float32 else 1e-2

    print("Main Output:")
    print(torch.allclose(expected, Z, atol=atol, rtol=rtol))
    print(torch.max(torch.abs(expected - Z)).item())
    print("\n")
    print("Spectral Output")
    print(torch.allclose(exp_spect, spect, atol=atol, rtol=rtol))
    print(torch.max(torch.abs(exp_spect - spect)).item())

    # now check the backward pass and grads
    dO = torch.ones_like(expected)
    dS = torch.ones_like(exp_spect)
    torch.autograd.backward([expected, exp_spect],[dO, dS])
    torch.autograd.backward([Z, spect],[dO, dS])

    print("\n")
    print("Grad D Output")
    print(torch.allclose(D_1.grad, D_2.grad, atol=atol, rtol=rtol))
    print(torch.max(torch.abs(D_1.grad - D_2.grad)).item())
    print("\n")
    print("Grad Y Output")
    print(torch.allclose(Y_1.grad, Y_2.grad, atol=atol, rtol=rtol))
    print(torch.max(torch.abs(Y_1.grad - Y_2.grad)).item())

    '''
    Main Output:
    True
    2.5331974029541016e-07


    Spectral Output
    True
    9.298324584960938e-06


    Grad D Output
    True
    0.000579833984375


    Grad Y Output
    True
    0.000244140625
    '''