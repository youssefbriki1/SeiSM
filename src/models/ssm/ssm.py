import torch
import torch.nn as nn
from mamba_ssm import Mamba


class SpatialSSM(nn.Module):
    """
    SSM module for spatial (ResNet) features.

    Input:  (batch, seq_len, d_input)   — seq_len = 10 history years
    Output: (batch, d_model)  OR  (batch, seq_len, d_model)
            depending on the pooling strategy chosen below.

    Integration note:
        In the merged pipeline, two SSMs run in parallel:
            spatial_vec   = SpatialSSM(resnet_out)     # this module
            waveform_vec  = WaveformSSM(waveform_feats)
        Then:
            merged = torch.cat([spatial_vec, waveform_vec], dim=-1)
            logits = mlp(merged)
        So d_model here and in WaveformSSM determine the MLP's input size.
    """

    def __init__(
        self,
        d_input: int,       # ResNet output width
        d_model: int = 128, # SSM hidden width
        d_state: int = 16,  # Mamba state expansion 
        n_layers: int = 2,  # number of stacked Mamba layers 
    ):
        super().__init__()

        # Projects ResNet output width into SSM hidden width
        self.proj_in = nn.Linear(d_input, d_model)

        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_input)
        Returns:
            pooled: (batch, d_model)   — if using options 1, 2, or 3
                OR
            sequence: (batch, seq_len, d_model) — if using option 4
        """
        x = self.proj_in(x)         # (batch, seq_len, d_model)

        for layer in self.layers:
            x = x + layer(x)        # residual around each Mamba block

        x = self.norm(x)            # (batch, seq_len, d_model)


        # Last timestep 
        return x[:, -1, :]
