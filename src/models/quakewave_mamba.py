import torch
import torch.nn as nn
try:
    from mamba_ssm import Mamba
    use_minimal = False
except ImportError:
    print("Warning: mamba_ssm not found. Using mamba_minimal fallback.")
    from .mamba_minimal import MambaBlock, ModelArgs
    use_minimal = True
class QuakeWaveMamba2(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        d_model: int = 256,  
        d_state: int = 64,  
        d_conv: int = 4,
        expand: int = 4,    
        headdim: int = 64, 
        n_layers: int = 8,  
        dropout: float = 0.2,
    ):
        """
        A Scaled-Up Mamba-2 architecture designed for 1D Seismic Waveform Regression.
        Expects input shape: (Batch, 3, 8192)
        Outputs shape: (Batch, 1) -> The predicted continuous magnitude.
        """
        super().__init__()
        
        # (Batch, 3, 8192) -> (Batch, d_model, 1024)
        self.cnn_embedder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=16, stride=4, padding=6),
            nn.GELU(),
            # Output channels of the CNN must match the new d_model
            nn.Conv1d(64, d_model, kernel_size=16, stride=2, padding=7),
            nn.GELU()
        )
        
        self.layers = nn.ModuleList([
            Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim
            ) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 128), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch, 3, 8192) raw waveforms
        """
        x = self.cnn_embedder(x)      # (Batch, d_model, 1024)
        x = x.transpose(1, 2).contiguous()         # (Batch, 1024, d_model) 
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        
        pooled = x.mean(dim=1)        # (Batch, d_model)
        
        magnitude = self.regressor(pooled) # (Batch, 1)
        
        return magnitude

if __name__ == "__main__":  
    model = QuakeWaveMamba2()
    print(model)