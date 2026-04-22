import torch.nn as nn
from mamba_ssm.modules.mamba2 import Mamba2


class QuakeMamba2(nn.Module):
    def __init__(self, d_model=128, d_state=16, input_dim=65*282, num_classes=4, num_patches=65, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.proj_in = nn.Linear(self.input_dim, d_model)
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
