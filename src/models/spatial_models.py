import torch
import torch.nn as nn
from .safenet_embeddings import SafeNetEmbeddings

class SafeNetFull(SafeNetEmbeddings):
    """Full SafeNet: Maps/Catalog embeddings → LSTM → ViT → classification.

    Follows the original architecture of Safenet (reproduced from paper):
      1. Fuse maps (32-d) + catalog (32-d) → 64-d per region per timestep
      2. LSTM across time for each region → 32-d time-aware embedding
      3. Separate LSTM for global token → 32-d
      4. Learnable position encoding + Vision Transformer encoder
      5. Linear classification head
    """

    def __init__(
        self,
        num_classes=4,
        map_channels=5,
        catalog_features=282,
        embed_dim=32,
        num_patches=64,
        num_heads=2,
        transformer_layers=1,
        dropout=0.2,
    ):
        super().__init__(
            num_classes=num_classes,
            map_channels=map_channels,
            catalog_features=catalog_features,
            embed_dim=embed_dim,
            num_patches=num_patches,
        )
        fused_dim = embed_dim * 2  # 64

        # Time-aware embedding (one LSTM per region, one for global token)
        self.regional_norm = nn.LayerNorm(fused_dim)
        self.global_norm   = nn.LayerNorm(embed_dim)
        self.regional_lstm = nn.LSTM(fused_dim, embed_dim, batch_first=True)
        self.global_lstm   = nn.LSTM(embed_dim, embed_dim, batch_first=True)

        # Position encoding (learnable, as in PatchEncoder)
        total_patches = num_patches + 1  # 86 = 85 regions + global token
        self.pos_proj  = nn.Linear(embed_dim, embed_dim)
        self.pos_embed = nn.Embedding(total_patches, embed_dim)

        # Vision Transformer encoder
        self.pre_transformer_norm = nn.LayerNorm(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )

        # Override the simple head from parent with one on embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, inputs):
        z, z_g = self._encode(inputs)  # z: (B,T,85,64), z_g: (B,T,32)
        B, T, P, D = z.shape

        # --- Layer norm ---
        z   = self.regional_norm(z)     # (B, T, 85, 64)
        z_g = self.global_norm(z_g)     # (B, T, 32)

        # --- Time-aware: LSTM per region ---
        z = z.permute(0, 2, 1, 3)                              # (B, 85, T, 64)
        z = z.reshape(B * P, T, D)                              # (B*85, T, 64)
        z, _ = self.regional_lstm(z)                            # (B*85, T, 32)
        z = z[:, -1, :]                                         # (B*85, 32) last step
        z = z.reshape(B, P, -1)                                 # (B, 85, 32)

        # --- Time-aware: LSTM for global token ---
        z_g, _ = self.global_lstm(z_g)                          # (B, T, 32)
        z_g = z_g[:, -1, :].unsqueeze(1)                        # (B, 1, 32)

        # --- Prepend global token ---
        z = torch.cat([z_g, z], dim=1)                          # (B, 86, 32)

        # --- Position encoding (additive) ---
        pos_ids = torch.arange(P + 1, device=z.device)
        pos_enc = self.pos_proj(z) + self.pos_embed(pos_ids)
        z = z + self.pre_transformer_norm(pos_enc)              # (B, 86, 32)

        # --- Vision Transformer ---
        z = self.transformer(z)                                 # (B, 86, 32)

        # --- Drop global token, classify ---
        z = z[:, 1:, :]                                         # (B, 85, 32)
        return self.head(z)                                     # (B, 85, 4)


# ---------------------------------------------------------------------------
# SafeNet SSM: embeddings + Mamba → classification
# ---------------------------------------------------------------------------

class SeiSM(SafeNetEmbeddings):
    """SafeNet with Mamba SSM replacing LSTM + ViT.

    Maps/Catalog embeddings are fused and flattened across patches into a
    single token per timestep.  A stacked residual Mamba processes the
    temporal sequence, and the final hidden state is projected to per-patch
    class logits.
    """

    def __init__(
        self,
        num_classes=4,
        map_channels=5,
        catalog_features=282,
        embed_dim=32,
        num_patches=64,
        d_model=128,
        d_state=16,
        n_ssm_layers=2,
    ):
        try:
            from mamba_ssm import Mamba
            use_minimal = False
        except ImportError:
            print("Warning: mamba_ssm not found. Using mamba_minimal fallback.")
            from .mamba_minimal import MambaBlock, ModelArgs
            use_minimal = True

        super().__init__(
            num_classes=num_classes,
            map_channels=map_channels,
            catalog_features=catalog_features,
            embed_dim=embed_dim,
            num_patches=num_patches,
        )
        fused_dim = embed_dim * 2  # 64
        input_dim = num_patches * fused_dim + embed_dim  # P*64 + 32

        self.regional_norm = nn.LayerNorm(fused_dim)
        self.global_norm = nn.LayerNorm(embed_dim)
        self.proj_in = nn.Linear(input_dim, d_model)
        
        if use_minimal:
            # Create a dummy ModelArgs (n_layer and vocab_size aren't used by MambaBlock)
            args = ModelArgs(d_model=d_model, d_state=d_state, d_conv=4, expand=2, n_layer=1, vocab_size=1)
            self.ssm_layers = nn.ModuleList([
                MambaBlock(args) for _ in range(n_ssm_layers)
            ])
        else:
            self.ssm_layers = nn.ModuleList([
                Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
                for _ in range(n_ssm_layers)
            ])

        self.ssm_norm = nn.LayerNorm(d_model)

        # Override parent's head: project from d_model to all patch logits
        self._num_classes = num_classes
        self.head = nn.Linear(d_model, num_patches * num_classes)

    def forward(self, inputs):
        z, z_g = self._encode(inputs)   # z: (B,T,P,64), z_g: (B,T,32)
        B, T, P, D = z.shape

        # --- Norm ---
        z = self.regional_norm(z)       # (B, T, P, 64)
        z_g = self.global_norm(z_g)     # (B, T, 32)

        # --- Flatten patches into a single token per timestep ---
        z = z.reshape(B, T, P * D)                              # (B, T, P*64)
        z = torch.cat([z, z_g], dim=-1)                         # (B, T, P*64+32)

        # --- Project + Mamba ---
        z = self.proj_in(z)                                     # (B, T, d_model)
        for layer in self.ssm_layers:
            z = z + layer(z)                                    # residual
        z = self.ssm_norm(z)                                    # (B, T, d_model)
        z = z[:, -1, :]                                         # (B, d_model)

        # --- Classify ---
        logits = self.head(z)                                   # (B, P*num_classes)
        return logits.reshape(B, P, self._num_classes)          # (B, P, num_classes)
