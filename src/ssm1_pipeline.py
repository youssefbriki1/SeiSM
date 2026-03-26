#!/usr/bin/env python3
"""
main.py — SafeNet-like model pipeline entry point.

Produces data/ssm1_output.pickle — input for the MLP fusion layer.

Pipeline:
  1. Preprocessing  (run_preprocessing.sh)
  2. Load data      (MultimodalSafeNetDataset)
  3. Embedding      (SafeNetEmbeddings._encode)
  4. SpatialSSM     → (B, 85, 128)
  5. Save           → data/ssm1_output.pickle
"""

import os
from pathlib import Path
import sys
import pickle
import subprocess
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from models.ssm import SpatialSSM
from models.safenet_embeddings import SafeNetEmbeddings
from utils.dataset import MultimodalSafeNetDataset

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_DIM    = 32
FUSED_DIM    = EMBED_DIM * 2   # 64

SEQ_LEN      = 10
NUM_PATCHES  = 85

SSM_D_MODEL  = 128
SSM_D_STATE  = 16
SSM_N_LAYERS = 2

NUM_CLASSES  = 4

DATA_DIR             = Path(__file__).parent.parent / 'data'
TRAINING_PICKLE      = os.path.join(DATA_DIR, "training_output.pickle")
TESTING_PICKLE       = os.path.join(DATA_DIR, "testing_output.pickle")
TRAINING_LABELS      = os.path.join(DATA_DIR, "training_labels.pickle")
TESTING_LABELS       = os.path.join(DATA_DIR, "testing_labels.pickle")
SSM1_OUTPUT_PICKLE   = os.path.join(DATA_DIR, "ssm1_output.pickle")
PREPROCESSING_SCRIPT = os.path.join(Path(__file__).parent, "data-processing/safenet/run_preprocessing.sh")


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test_spatial_ssm():
    print("=" * 60)
    print("Smoke test: SpatialSSM (per-patch, option B)")
    print("=" * 60)

    model = SpatialSSM(
        d_input  = FUSED_DIM,
        d_model  = SSM_D_MODEL,
        d_state  = SSM_D_STATE,
        n_layers = SSM_N_LAYERS,
    ).to(DEVICE)

    batch = 2
    fake_fused = torch.randn(batch, SEQ_LEN, NUM_PATCHES, FUSED_DIM, device=DEVICE)

    B, T, P, D = fake_fused.shape
    x = fake_fused.permute(0, 2, 1, 3).reshape(B * P, T, D)  # (B*85, T, 64)
    out = model(x).reshape(B, P, SSM_D_MODEL)                 # (B, 85, 128)

    print(f"  Input  shape : {fake_fused.shape}")
    print(f"  Output shape : {out.shape}    <- (batch, patches, d_model)")
    print(f"  Device       : {DEVICE}")
    print(f"  Params       : {sum(p.numel() for p in model.parameters()):,}")
    print()


# ═══════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline(skip_preprocessing=False, validate=False):
    if not skip_preprocessing:
        print("=" * 60)
        print("Step 1: Running preprocessing script...")
        print("=" * 60)
        cmd = ["bash", str(PREPROCESSING_SCRIPT)]
        if validate:
            print("Adding preprocessing validation...")
            cmd.append("--validate")
        subprocess.run(cmd, check=True)

    # ── Step 2: Load data ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Loading data...")
    print("=" * 60)

    train_dataset = MultimodalSafeNetDataset(TRAINING_PICKLE, TRAINING_LABELS)
    test_dataset  = MultimodalSafeNetDataset(TESTING_PICKLE,  TESTING_LABELS)

    # batch_size=1 since each sample is already a full year's data
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False)

    print(f"  Training samples : {len(train_dataset)}")
    print(f"  Testing  samples : {len(test_dataset)}")

    # ── Step 3: Instantiate models ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Instantiating embedder + SSM...")
    print("=" * 60)

    embedder = SafeNetEmbeddings(
        num_classes      = NUM_CLASSES,
        map_channels     = 5,
        catalog_features = 282,
        embed_dim        = EMBED_DIM,
        num_patches      = NUM_PATCHES,
    ).to(DEVICE)

    spatial_ssm = SpatialSSM(
        d_input  = FUSED_DIM,
        d_model  = SSM_D_MODEL,
        d_state  = SSM_D_STATE,
        n_layers = SSM_N_LAYERS,
    ).to(DEVICE)

    embedder.eval()
    spatial_ssm.eval()

    print(f"  Embedder params  : {sum(p.numel() for p in embedder.parameters()):,}")
    print(f"  SSM params       : {sum(p.numel() for p in spatial_ssm.parameters()):,}")

    # ── Step 4: Run forward pass, collect SSM outputs ─────────────────
    print("\n" + "=" * 60)
    print("Step 4: Running embedder + SSM...")
    print("=" * 60)

    def process_loader(loader, split_name):
        outputs = []
        labels  = []

        with torch.no_grad():
            for i, (inputs, label) in enumerate(loader):
                catalog = inputs["catalog"].to(DEVICE)  # (B, 10, 86, 282)
                maps    = inputs["maps"].to(DEVICE)     # (B, 10, 85, 50, 50, 5)

                # Embed
                z, _ = embedder._encode({
                    "catalog": catalog,
                    "maps":    maps,
                })                                      # (B, T, 85, 64)

                # SSM (per-patch)
                B, T, P, D = z.shape
                x = z.permute(0, 2, 1, 3).reshape(B * P, T, D)
                ssm_out = spatial_ssm(x).reshape(B, P, SSM_D_MODEL)  # (B, 85, 128)

                outputs.append(ssm_out.cpu())
                labels.append(label)

                if (i + 1) % 5 == 0:
                    print(f"  [{split_name}] {i+1}/{len(loader)} samples processed")

        return torch.cat(outputs, dim=0), torch.cat(labels, dim=0)

    train_out, train_labels = process_loader(train_loader, "train")
    test_out,  test_labels  = process_loader(test_loader,  "test")

    print(f"\n  Train SSM output shape : {train_out.shape}   <- (samples, 85, 128)")
    print(f"  Test  SSM output shape : {test_out.shape}")

    # ── Step 5: Save to pickle ────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Step 5: Saving SSM output to {SSM1_OUTPUT_PICKLE}...")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)
    output = {
        "train": {
            "ssm1_out": train_out.numpy(),    # (N_train, 85, 128)
            "labels":   train_labels.numpy()  # (N_train, 85)
        },
        "test": {
            "ssm1_out": test_out.numpy(),     # (N_test, 85, 128)
            "labels":   test_labels.numpy()   # (N_test, 85)
        },
        "config": {
            "d_model":     SSM_D_MODEL,
            "num_patches": NUM_PATCHES,
            "num_classes": NUM_CLASSES,
        }
    }

    with open(SSM1_OUTPUT_PICKLE, "wb") as f:
        pickle.dump(output, f)

    print(f"  Saved -> {SSM1_OUTPUT_PICKLE}")
    print("\nDone. SSM1 output ready for MLP fusion.")


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")

    # Smoke test always runs
    smoke_test_spatial_ssm()

    # Full pipeline — set skip_preprocessing=True if pickles already exist
    run_pipeline(skip_preprocessing=False)