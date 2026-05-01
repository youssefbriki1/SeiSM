import torch
import torch.nn as nn

class SeiSM(nn.Module):
    def __init__(
        self, 
        mamba_model: nn.Module, 
        safenet_model: nn.Module, 
        mlp_hidden_dim: int = 32, 
        final_classes: int = 4
    ):
        """
        Merges Mamba (Waveform) and SafeNet (Multimodal) using an MLP. Base models are frozen.
        """
        super().__init__()
        
        self.mamba = mamba_model
        self.safenet = safenet_model
        
        for param in self.mamba.parameters():
            param.requires_grad = False
            
        for param in self.safenet.parameters():
            param.requires_grad = False
            
        self.mamba.eval()
        self.safenet.eval()

        # SafeNetSSM outputs (B, P, 4) logits. Mamba outputs (B, 1) magnitude.
        fusion_in_features = final_classes + 1 
        
        self.mlp = nn.Sequential(
            nn.Linear(fusion_in_features, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden_dim, final_classes)
        )

    def forward(self, waveforms: torch.Tensor, safenet_inputs: dict) -> torch.Tensor:
        with torch.no_grad():
            mamba_out = self.mamba(waveforms)          # (B, 1) or (B,)
            
            safenet_out = self.safenet(safenet_inputs) # (B, Patches, 4)
        
        # Ensure mamba_out is (B, 1)
        if mamba_out.dim() == 1:
            mamba_out = mamba_out.unsqueeze(-1)
        
        Batch, Patches, _ = safenet_out.shape
        
        # Expand Mamba output across all patches
        mamba_expanded = mamba_out.unsqueeze(1).expand(Batch, Patches, 1)
        
        # Concatenate features -> (B, Patches, 5)
        fused_features = torch.cat([safenet_out, mamba_expanded], dim=-1)
        
        # Pass through the MLP to get final logits -> (B, Patches, 4)
        final_output = self.mlp(fused_features)
        
        return final_output
    
    def train(self, mode=True):
        """
        Override the train method to ensure base models stay in eval mode
        even when fused_model.train() is called.
        """
        super().train(mode)
        if mode:
            self.mamba.eval()
            self.safenet.eval()