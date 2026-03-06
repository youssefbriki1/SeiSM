import torch.nn as nn
from mamba_ssm import Mamba2


class QuakeMamba2(nn.Module):
    def __init__(self, d_model=1024, d_state=16, input_dim=64, num_classes=4, num_patches=16, **kwargs):
        super().__init__()
        self.input_dim = input_dim

        # Multi-step projection: gradually compress input_dim → d_model
        # instead of a single extreme compression step
        mid1 = max(d_model, input_dim // 4)
        mid2 = max(d_model, input_dim // 8)
        self.proj_in = nn.Sequential(
            nn.Linear(self.input_dim, mid1),
            nn.LayerNorm(mid1),
            nn.GELU(),
            nn.Linear(mid1, mid2),
            nn.LayerNorm(mid2),
            nn.GELU(),
            nn.Linear(mid2, d_model),
            nn.LayerNorm(d_model),
        )

        mamba_kwargs = {"use_mem_eff_path": False}
        mamba_kwargs.update(kwargs)
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, **mamba_kwargs)
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.proj_out = nn.Linear(d_model, self.num_classes * self.num_patches)
        
    def forward(self, x):
        B, L, P, F = x.shape
        x = x.reshape(B, L, P * F).contiguous()
        x = self.proj_in(x).contiguous()
        x = self.mamba(x)
        
        last_step = x[:, -1, :] 
        
        logits = self.proj_out(last_step)
        
        return logits.view(B, self.num_patches, self.num_classes)
