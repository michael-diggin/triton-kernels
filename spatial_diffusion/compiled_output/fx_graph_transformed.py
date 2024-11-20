class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[2048, 128]", arg1_1: "bf16[2048]", arg2_1: "bf16[2048, 256]", arg3_1: "bf16[128]", arg4_1: "bf16[256]"):
         # File: /content/temp.py:11 in torch_diffusion, code: diffs = torch.exp(-evalues.unsqueeze(-1)*times)
        unsqueeze_1: "bf16[128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        neg: "bf16[128, 1]" = torch.ops.aten.neg.default(unsqueeze_1);  unsqueeze_1 = None
        mul_1: "bf16[128, 256]" = torch.ops.aten.mul.Tensor(neg, arg4_1);  neg = arg4_1 = None
        exp: "bf16[128, 256]" = torch.ops.aten.exp.default(mul_1);  mul_1 = None
        
         # File: /content/temp.py:8 in torch_diffusion, code: b_t = basis.transpose(-2, -1)
        permute: "bf16[128, 2048]" = torch.ops.aten.permute.default(arg0_1, [1, 0])
        
         # File: /content/temp.py:9 in torch_diffusion, code: x_m = x*mass.unsqueeze(-1)
        unsqueeze: "bf16[2048, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, -1);  arg1_1 = None
        mul: "bf16[2048, 256]" = torch.ops.aten.mul.Tensor(arg2_1, unsqueeze);  arg2_1 = unsqueeze = None
        
         # File: /content/temp.py:10 in torch_diffusion, code: in_basis = torch.matmul(b_t, x_m)
        mm: "bf16[128, 256]" = torch.ops.aten.mm.default(permute, mul);  permute = mul = None
        
         # File: /content/temp.py:12 in torch_diffusion, code: spectral = diffs*in_basis
        mul_2: "bf16[128, 256]" = torch.ops.aten.mul.Tensor(exp, mm);  exp = mm = None
        
         # File: /content/temp.py:13 in torch_diffusion, code: return torch.matmul(basis, spectral), spectral
        mm_1: "bf16[2048, 256]" = torch.ops.aten.mm.default(arg0_1, mul_2);  arg0_1 = None
        return (mm_1, mul_2)
        