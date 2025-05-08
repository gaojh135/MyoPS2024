import torch
from torch import autocast, nn
from mamba_ssm import Mamba2

class Mamba2Layer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, headdim = 8):
        super().__init__()
        print(f"Mamba2Layer: dim: {dim}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba2(
                d_model=dim,      # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                headdim=headdim,
        )

    @autocast(device_type='cuda', enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            x = x.to(torch.float32)

        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)

        return out