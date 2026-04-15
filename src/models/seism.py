import torch 
import torch.nn as nn
import torch.nn.functional as F
from quakewave_mamba import QuakeWaveMamba2

class Seism(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_layers=4, hidden_dim=64, freeze_mamba=True):
        super(Seism, self).__init__()
        self.mamba = QuakeWaveMamba2(in_channels, out_channels, num_layers, hidden_dim)
        if freeze_mamba:
            for param in self.mamba.parameters():
                param.requires_grad = False
                
        self.lin1 = nn.Linear(2, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, out_channels)
        
    def forward(self, x):
        mamba_out = self.mamba(x) 
        
        # Take the last time step from mamba output
        last_time_step = mamba_out[:, :, -1]  # shape: (batch_size, out_channels)
        
        # Pass through linear layers
        out = F.relu(self.lin1(last_time_step))  # shape: (batch_size, hidden_dim)
        out = F.relu(self.lin2(out))  # shape: (batch_size, hidden_dim)
        out = self.lin3(out)  # shape: (batch_size, out_channels)
        
        return out