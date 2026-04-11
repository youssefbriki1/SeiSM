#!/usr/bin/env python3
"""
pipeline.py — SafeNet-like model pipeline, dataset-agnostic.

Usage:
    pipeline.smoke_test()   # sanity check shapes
    pipeline.train()        # train end-to-end, save checkpoint + print metrics
    pipeline.run()          # load checkpoint, save SSM outputs for MLP fusion
"""

import os
import time
import pickle
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from models.ssm import SpatialSSM
from models.safenet_embeddings import SafeNetEmbeddings
from utils.dataset import MultimodalSafeNetDataset
from utils.focal_loss import FocalLoss

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM    = 32
FUSED_DIM    = EMBED_DIM * 2   # 64
SEQ_LEN      = 10
SSM_D_MODEL  = 128
SSM_D_STATE  = 16
SSM_N_LAYERS = 2
NUM_CLASSES  = 4
CLASS_NAMES  = ["M<5", "M5-6", "M6-7", "M≥7"]


# ═══════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════

class SafeNetPipeline:

    def __init__(
        self,
        data_dir,
        preprocessing_script,
        num_patches,
        training_pickle    = "training_output.pickle",
        testing_pickle     = "testing_output.pickle",
        training_labels    = "training_labels.pickle",
        testing_labels     = "testing_labels.pickle",
        ssm1_output_pickle = "ssm1_output.pickle",
        num_epochs         = 20,
        learning_rate      = 1e-4,
        class_weights      = None,  # e.g. torch.tensor([1.0, 4.0, 15.0, 78.0])
    ):
        self.data_dir             = Path(data_dir)
        self.preprocessing_script = preprocessing_script
        self.num_patches          = num_patches
        self.num_epochs           = num_epochs
        self.learning_rate        = learning_rate
        self.class_weights        = class_weights

        self.training_pickle    = self.data_dir / training_pickle
        self.testing_pickle     = self.data_dir / testing_pickle
        self.training_labels    = self.data_dir / training_labels
        self.testing_labels     = self.data_dir / testing_labels
        self.ssm1_output_pickle = self.data_dir / ssm1_output_pickle
        self.checkpoint_path    = self.data_dir / "checkpoint.pt"

    # ── Internal helpers ─────────────────────────────────────────────

    def _build_models(self):
        embedder = SafeNetEmbeddings(
            num_classes      = NUM_CLASSES,
            map_channels     = 5,
            catalog_features = 282,
            embed_dim        = EMBED_DIM,
            num_patches      = self.num_patches,
        ).to(DEVICE)

        spatial_ssm = SpatialSSM(
            d_input  = FUSED_DIM,
            d_model  = SSM_D_MODEL,
            d_state  = SSM_D_STATE,
            n_layers = SSM_N_LAYERS,
        ).to(DEVICE)

        head = nn.Linear(SSM_D_MODEL, NUM_CLASSES).to(DEVICE)

        return embedder, spatial_ssm, head

    def _load_checkpoint(self):
        if not self.checkpoint_path.exists():
            raise RuntimeError(
                f"No checkpoint found at {self.checkpoint_path}. Run train() first."
            )
        embedder, spatial_ssm, head = self._build_models()
        checkpoint = torch.load(self.checkpoint_path, map_location=DEVICE)
        embedder.load_state_dict(checkpoint["embedder"])
        spatial_ssm.load_state_dict(checkpoint["spatial_ssm"])
        head.load_state_dict(checkpoint["head"])
        return embedder, spatial_ssm, head

    def _forward(self, embedder, spatial_ssm, head, inputs):
        catalog = inputs["catalog"].to(DEVICE)
        maps    = inputs["maps"].to(DEVICE)

        z, _ = embedder._encode({"catalog": catalog, "maps": maps})

        B, T, P, D = z.shape
        x       = z.permute(0, 2, 1, 3).reshape(B * P, T, D)
        ssm_out = spatial_ssm(x).reshape(B, P, SSM_D_MODEL)  # (B, patches, 128)
        logits  = head(ssm_out)                               # (B, patches, 4)

        return logits, ssm_out

    def _get_loaders(self, shuffle_train=False):
        train_dataset = MultimodalSafeNetDataset(self.training_pickle, self.training_labels)
        test_dataset  = MultimodalSafeNetDataset(self.testing_pickle,  self.testing_labels)
        train_loader  = DataLoader(train_dataset, batch_size=1, shuffle=shuffle_train)
        test_loader   = DataLoader(test_dataset,  batch_size=1, shuffle=False)
        return train_loader, test_loader

    # ── Public API ───────────────────────────────────────────────────

    def smoke_test(self):
        """Sanity check: verify shapes are correct before touching real data."""
        print("=" * 60)
        print(f"Smoke test: SpatialSSM ({self.num_patches} patches)")
        print("=" * 60)

        model = SpatialSSM(
            d_input  = FUSED_DIM,
            d_model  = SSM_D_MODEL,
            d_state  = SSM_D_STATE,
            n_layers = SSM_N_LAYERS,
        ).to(DEVICE)

        batch      = 2
        fake_input = torch.randn(batch, SEQ_LEN, self.num_patches, FUSED_DIM, device=DEVICE)

        B, T, P, D = fake_input.shape
        x   = fake_input.permute(0, 2, 1, 3).reshape(B * P, T, D)
        out = model(x).reshape(B, P, SSM_D_MODEL)

        print(f"  Input  shape : {fake_input.shape}")
        print(f"  Output shape : {out.shape}  <- (batch, patches, d_model)")
        print(f"  Device       : {DEVICE}")
        print(f"  Params       : {sum(p.numel() for p in model.parameters()):,}")
        print()

    def train(self, skip_preprocessing=False):
        """Train embedder + SSM + head end-to-end. Saves checkpoint and prints metrics."""
        if not skip_preprocessing:
            print("=" * 60)
            print("Step 1: Running preprocessing...")
            print("=" * 60)
            subprocess.run(["bash", str(self.preprocessing_script)], check=True)

        print("\n" + "=" * 60)
        print("Step 2: Loading data...")
        print("=" * 60)
        train_loader, test_loader = self._get_loaders(shuffle_train=True)
        print(f"  Training samples : {len(train_loader)}")
        print(f"  Testing  samples : {len(test_loader)}")

        print("\n" + "=" * 60)
        print("Step 3: Building models...")
        print("=" * 60)
        embedder, spatial_ssm, head = self._build_models()
        print(f"  Embedder params  : {sum(p.numel() for p in embedder.parameters()):,}")
        print(f"  SSM params       : {sum(p.numel() for p in spatial_ssm.parameters()):,}")
        print(f"  Head params      : {sum(p.numel() for p in head.parameters()):,}")

        params    = (list(embedder.parameters()) +
                     list(spatial_ssm.parameters()) +
                     list(head.parameters()))
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        weights   = self.class_weights.to(DEVICE) if self.class_weights is not None else None
        criterion = FocalLoss(alpha=weights, gamma=2.0)

        print("\n" + "=" * 60)
        print(f"Step 4: Training for {self.num_epochs} epochs...")
        print("=" * 60)
        train_start = time.time()

        for epoch in range(self.num_epochs):
            embedder.train()
            spatial_ssm.train()
            head.train()

            epoch_loss = 0.0
            for inputs, labels in train_loader:
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                logits, _ = self._forward(embedder, spatial_ssm, head, inputs)
                loss = criterion(
                    logits.reshape(-1, NUM_CLASSES),
                    labels.reshape(-1),
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"  Epoch {epoch+1}/{self.num_epochs}  loss: {epoch_loss/len(train_loader):.4f}")

        training_time = time.time() - train_start
        print(f"\n  Training time: {training_time:.1f}s")

        print("\n" + "=" * 60)
        print("Step 5: Saving checkpoint...")
        print("=" * 60)
        torch.save({
            "embedder":    embedder.state_dict(),
            "spatial_ssm": spatial_ssm.state_dict(),
            "head":        head.state_dict(),
        }, self.checkpoint_path)
        print(f"  Saved -> {self.checkpoint_path}")

        print("\n" + "=" * 60)
        print("Step 6: Evaluating on test set...")
        print("=" * 60)
        self.evaluate(embedder, spatial_ssm, head, test_loader)

    def evaluate(self, embedder, spatial_ssm, head, test_loader):
        """Compute and print all metrics on the test set."""
        embedder.eval()
        spatial_ssm.eval()
        head.eval()

        all_preds  = []
        all_labels = []

        infer_start = time.time()
        with torch.no_grad():
            for inputs, labels in test_loader:
                logits, _ = self._forward(embedder, spatial_ssm, head, inputs)
                preds = logits.argmax(dim=-1)
                all_preds.append(preds.cpu().reshape(-1))
                all_labels.append(labels.reshape(-1))
        infer_time = time.time() - infer_start

        all_preds  = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        acc         = accuracy_score(all_labels, all_preds)
        f1_macro    = f1_score(all_labels, all_preds, average='macro',    zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        per_precision, per_recall, per_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, labels=list(range(NUM_CLASSES)), zero_division=0
        )

        print(f"  Accuracy         : {acc:.4f}")
        print(f"  Macro F1         : {f1_macro:.4f}")
        print(f"  Weighted F1      : {f1_weighted:.4f}")
        print(f"  Inference time   : {infer_time:.2f}s")
        print()
        print("  Per-class breakdown:")
        print(f"  {'Class':8s}  {'Precision':>10s}  {'Recall':>8s}  {'F1':>8s}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name:8s}  {per_precision[i]:>10.4f}  {per_recall[i]:>8.4f}  {per_f1[i]:>8.4f}")

    def run(self):
        """
        Load trained checkpoint and save SSM outputs to pickle for MLP fusion.
        Must call train() first.
        """
        print("=" * 60)
        print("Loading trained checkpoint...")
        print("=" * 60)
        embedder, spatial_ssm, head = self._load_checkpoint()
        embedder.eval()
        spatial_ssm.eval()

        train_loader, test_loader = self._get_loaders(shuffle_train=False)

        def process_loader(loader, split_name):
            outputs = []
            labels  = []
            with torch.no_grad():
                for i, (inputs, label) in enumerate(loader):
                    _, ssm_out = self._forward(embedder, spatial_ssm, head, inputs)
                    outputs.append(ssm_out.cpu())
                    labels.append(label)
                    if (i + 1) % 5 == 0:
                        print(f"  [{split_name}] {i+1}/{len(loader)} samples processed")
            return torch.cat(outputs, dim=0), torch.cat(labels, dim=0)

        print("\nGenerating SSM outputs...")
        train_out, train_labels = process_loader(train_loader, "train")
        test_out,  test_labels  = process_loader(test_loader,  "test")

        print(f"\n  Train output shape : {train_out.shape}")
        print(f"  Test  output shape : {test_out.shape}")

        os.makedirs(self.data_dir, exist_ok=True)
        output = {
            "train": {
                "ssm1_out": train_out.numpy(),
                "labels":   train_labels.numpy(),
            },
            "test": {
                "ssm1_out": test_out.numpy(),
                "labels":   test_labels.numpy(),
            },
            "config": {
                "d_model":     SSM_D_MODEL,
                "num_patches": self.num_patches,
                "num_classes": NUM_CLASSES,
            },
        }

        with open(self.ssm1_output_pickle, "wb") as f:
            pickle.dump(output, f)

        print(f"\n  Saved -> {self.ssm1_output_pickle}")
        print("Done. SSM outputs ready for MLP fusion.")