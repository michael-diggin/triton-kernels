# triton-kernels

A repository holding some kernels I've been playing about with.

## Diffusion Net

This is a model architecture from https://arxiv.org/abs/2012.00888 and works pretty well on
3D mesh data.

Kernels for first two layers can be found here:
- [spatial_diffusion](spatial_diffusion/README.md)
- [spatial_gradients](spatial_gradients/README.md)

## Split-K Matmul

This is a matrix muliplication algorithm for increasing parallelism by computing
partial sums along the inner reduced dimension of the matmul.

- [split_k](split_k/README.md)
