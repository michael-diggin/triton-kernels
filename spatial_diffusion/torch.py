import torch

def torch_diffusion(x, basis, mass, evalues, times):
  # Concise implementation of spatial diffusion
  # Converts the input into the eigenbasis
  # Applies the diffusion times
  # Converts back to original basis
  b_t = basis.transpose(-2, -1)
  x_m = x*mass.unsqueeze(-1)
  in_basis = torch.matmul(b_t, x_m)
  diffs = torch.exp(-evalues.unsqueeze(-1)*times)
  spectral = diffs*in_basis
  return torch.matmul(basis, spectral), spectral