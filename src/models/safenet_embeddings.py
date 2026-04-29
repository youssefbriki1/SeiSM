"""
SafeNet Maps Embedding + Catalog Embedding in PyTorch.

Faithful port of the architecture from the TensorFlow implementation in
src/models/safenet/safenet.py (_init_graph, identity_block, convolutional_block).

Paper: Hu et al., "Scalable intermediate-term earthquake forecasting with
multimodal fusion neural networks", Scientific Reports 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Bottleneck blocks (match the TF identity_block / convolutional_block)
# ---------------------------------------------------------------------------

class IdentityBlock(nn.Module):
    """Bottleneck residual block with equal input/output dimensions."""

    def __init__(self, in_channels, filters, kernel_size=3):
        super().__init__()
        f1, f2, f3 = filters
        pad = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, f1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(f1)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm2d(f2)
        self.conv3 = nn.Conv2d(f2, f3, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(f3)

    def forward(self, x):
        shortcut = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + shortcut)


class ConvBlock(nn.Module):
    """Bottleneck block that changes spatial size and/or channel count."""

    def __init__(self, in_channels, filters, kernel_size=3, stride=2):
        super().__init__()
        f1, f2, f3 = filters
        pad = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, f1, 1, stride=stride, bias=False)
        self.bn1   = nn.BatchNorm2d(f1)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm2d(f2)
        self.conv3 = nn.Conv2d(f2, f3, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(f3)

        self.shortcut_conv = nn.Conv2d(in_channels, f3, 1, stride=stride, bias=False)
        self.shortcut_bn   = nn.BatchNorm2d(f3)

    def forward(self, x):
        shortcut = self.shortcut_bn(self.shortcut_conv(x))
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + shortcut)


# ---------------------------------------------------------------------------
# Maps Embedding (ResNet-style CNN)
# ---------------------------------------------------------------------------

class MapsEncoder(nn.Module):
    """Encodes a (5, 50, 50) map patch into a 32-d vector.

    Architecture (from safenet.py _init_graph lines 479-507):
        Conv 7×7 s2  → BN → ReLU → MaxPool 3×3 s2
        ConvBlock  [16,16,64]  s=1
        IdentityBlock [16,16,64] ×2
        ConvBlock  [32,32,128] s=2
        IdentityBlock [32,32,128] ×3
        AvgPool 2×2 s2 → Flatten → Linear(32) + ReLU
    """

    def __init__(self, in_channels=5, embed_dim=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # Stage 2: 32 → 64 channels, spatial unchanged (stride=1)
        self.stage2 = nn.Sequential(
            ConvBlock(32,  [16, 16, 64], stride=1),
            IdentityBlock(64, [16, 16, 64]),
            IdentityBlock(64, [16, 16, 64]),
        )

        # Stage 3: 64 → 128 channels, spatial /2
        self.stage3 = nn.Sequential(
            ConvBlock(64,  [32, 32, 128], stride=2),
            IdentityBlock(128, [32, 32, 128]),
            IdentityBlock(128, [32, 32, 128]),
            IdentityBlock(128, [32, 32, 128]),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(128, embed_dim)

    def forward(self, x):
        """x: (B, in_channels, 50, 50) → (B, embed_dim)"""
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return F.relu(self.fc(x))


# ---------------------------------------------------------------------------
# Catalog Embedding (4 × FC 32)
# ---------------------------------------------------------------------------

class CatalogEncoder(nn.Module):
    """Encodes a 282-d catalog feature vector into a 32-d vector.

    4 fully-connected layers with LeakyReLU, matching the TF implementation
    (safenet.py denselayer=[32,32,32,32]).
    """

    def __init__(self, in_features=282, embed_dim=32, num_layers=4):
        super().__init__()
        layers = []
        dim_in = in_features
        for _ in range(num_layers):
            layers.append(nn.Linear(dim_in, embed_dim))
            layers.append(nn.LeakyReLU(inplace=True))
            dim_in = embed_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: (B, 282) → (B, embed_dim)"""
        return self.net(x)


# ---------------------------------------------------------------------------
# Full model: Maps + Catalog embeddings → classification head
# ---------------------------------------------------------------------------

class SafeNetEmbeddings(nn.Module):
    """Two-tower SafeNet encoder with a simple classification head.

    Accepts a dict with keys ``catalog`` (10, 86, 282) and ``maps``
    (10, 85, 50, 50, 5).  Returns logits (85, num_classes).

    Temporal aggregation is mean-pooling over the 10 history years.
    For the full SafeNet temporal/spatial model see ``SafeNetFull``.
    """

    def __init__(
        self,
        num_classes=4,
        map_channels=5,
        catalog_features=282,
        embed_dim=32,
        num_patches=64,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.maps_enc    = MapsEncoder(in_channels=map_channels, embed_dim=embed_dim)
        self.catalog_enc = CatalogEncoder(in_features=catalog_features, embed_dim=embed_dim)
        self.norm        = nn.LayerNorm(embed_dim * 2)
        self.head        = nn.Linear(embed_dim * 2, num_classes)

    def _encode(self, inputs):
        """Shared encoding logic: returns fused (B, T, P, 2*embed_dim) and global (B, T, embed_dim)."""
        catalog = inputs["catalog"]       # (B, T, 86, 282)
        maps    = inputs["maps"]          # (B, T, 85, H, W, C)
        B, T = catalog.shape[:2]
        P = self.num_patches              # 85

        # --- Maps branch ---
        maps_flat = maps.reshape(B * T * P, *maps.shape[3:])   # (B*T*85, 50, 50, 5)
        maps_flat = maps_flat.permute(0, 3, 1, 2)              # (B*T*85, 5, 50, 50)
        z_m = self.maps_enc(maps_flat).reshape(B, T, P, -1)    # (B, T, 85, 32)

        # --- Catalog branch ---
        cat_global   = catalog[:, :, 0, :]                      # (B, T, 282)
        cat_regional = catalog[:, :, 1:, :]                     # (B, T, 85, 282)
        z_c = self.catalog_enc(
            cat_regional.reshape(B * T * P, -1)
        ).reshape(B, T, P, -1)                                  # (B, T, 85, 32)
        z_g = self.catalog_enc(
            cat_global.reshape(B * T, -1)
        ).reshape(B, T, -1)                                     # (B, T, 32)

        z = torch.cat([z_m, z_c], dim=-1)                      # (B, T, 85, 64)
        return z, z_g

    def forward(self, inputs):
        z, _ = self._encode(inputs)                             # (B, T, 85, 64)
        z = self.norm(z)
        z = z.mean(dim=1)                                       # (B, 85, 64)
        return self.head(z)                                     # (B, 85, 4)
