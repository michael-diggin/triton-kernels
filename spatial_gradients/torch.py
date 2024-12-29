import torch

def torch_spatial_gradients(x, grads_real, grads_im, linear_real, linear_im):
  x_grads_real = torch.mm(grads_real, x)
  x_grads_im = torch.mm(grads_im, x)
  x_real = x_grads_real @ linear_real.T - x_grads_im @ linear_im.T
  x_im = x_grads_im @ linear_real.T + x_grads_real @ linear_im.T
  x_dots = x_grads_real*x_real + x_grads_im*x_im
  return torch.tanh(x_dots)
 