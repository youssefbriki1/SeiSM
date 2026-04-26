"""
config.py — Shared hyperparameter config for SafeNetPipeline runs.

Import this in any runner script:
    from config import PipelineConfig
"""

from dataclasses import dataclass, asdict, field
from typing import Optional
import torch


@dataclass
class PipelineConfig:
    # ── Data ──────────────────────────────────────────────────────────────
    num_patches: int = 64

    # ── Training ──────────────────────────────────────────────────────────
    num_epochs:    int   = 20
    learning_rate: float = 1e-4
    batch_size:    int   = 1       

    # ── Loss ──────────────────────────────────────────────────────────────
    focal_gamma:   float = 2.0
    class_weights: Optional[list] = None  # e.g. [1.0, 4.0, 15.0, 78.0]

    # ── SSM architecture ──────────────────────────────────────────────────
    ssm_d_model:   int = 128
    ssm_d_state:   int = 16
    ssm_n_layers:  int = 2

    # ── W&B ───────────────────────────────────────────────────────────────
    wandb_project:  str  = "safenet-pipeline"
    wandb_run_name: str  = ""
    wandb_mode:     str  = "online"   # "online" | "offline" | "disabled"
    wandb_log_freq: int  = 50         # log every N training batches
    disable_wandb:  bool = False

    def to_dict(self) -> dict:
        """Serialise to plain dict (ready for wandb.init(config=...))."""
        return asdict(self)

    def class_weights_tensor(self) -> Optional[torch.Tensor]:
        if self.class_weights is None:
            return None
        return torch.tensor(self.class_weights, dtype=torch.float32)