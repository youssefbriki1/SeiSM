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


        # ── Option 1: Last timestep ────────────────────────────────────
        # Best default for causal SSMs (Mamba is causal): the final hidden
        # state has theoretically seen all 10 years in order.
        # Output: (batch, d_model)
        # Merge:  torch.cat([spatial_vec, waveform_vec], dim=-1)
        return x[:, -1, :]

        # ── Option 2: Mean pool ────────────────────────────────────────
        # More robust when the predictive signal is spread across years
        # rather than concentrated in recent history.
        # Output: (batch, d_model)
        # Merge:  torch.cat([spatial_vec, waveform_vec], dim=-1)
        # return x.mean(dim=1)

        # ── Option 3: Max pool ─────────────────────────────────────────
        # Captures peak activation across the sequence — useful when
        # rare high-magnitude years dominate the prediction.
        # Output: (batch, d_model)
        # Merge:  torch.cat([spatial_vec, waveform_vec], dim=-1)
        # return x.max(dim=1).values

        # ── Option 4: Full sequence ────────────────────────────────────
        # Passes the entire sequence to the merge layer. Requires the
        # waveform SSM to also return (batch, 10, d_model), and the MLP
        # to accept (batch, 10, d1+d2) — i.e. a flatten or attention head
        # before the final linear layers.
        # Only use this if both SSMs and the MLP team agree on it.
        # Output: (batch, seq_len, d_model)
        # Merge:  torch.cat([spatial_seq, waveform_seq], dim=-1)
        #         then flatten or pool before MLP
        # return x