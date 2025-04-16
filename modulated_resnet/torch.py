import torch

def modresnet_fwd(mod_act: torch.Tensor, syn_act: torch.Tensor, mod_weight: torch.Tensor, syn_weight: torch.Tensor):
  mod_act = torch.matmul(mod_act, mod_weight.transpose(-1, -2))
  mod_act = torch.nn.Sigmoid()(mod_act)
  syn_act = torch.matmul(syn_act, syn_weight.transpose(-1, -2))
  syn_act = torch.nn.Sigmoid()(syn_act)
  syn_act = syn_act*mod_act
  return mod_act, syn_act