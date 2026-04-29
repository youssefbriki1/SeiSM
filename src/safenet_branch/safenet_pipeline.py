"""
safenet_pipeline.py — SafeNet-like model pipeline

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
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from models.spatial_models import SeiSM
from utils.dataset import MultimodalSafeNetDataset
from utils.focal_loss import FocalLoss

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS 
# ═══════════════════════════════════════════════════════════════════════

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM   = 32
FUSED_DIM   = EMBED_DIM * 2   # 64
SEQ_LEN     = 10
NUM_CLASSES = 4
CLASS_NAMES = ["M<5", "M5-6", "M6-7", "M≥7"]


# ═══════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════

class SafeNetPipeline:

    def __init__(
        self,
        data_dir,
        preprocessing_script,
        num_patches,
        # ── Pickles ────────────────────────────────────────────────
        training_pickle    = "training_output.pickle",
        testing_pickle     = "testing_output.pickle",
        training_labels    = "training_labels.pickle",
        testing_labels     = "testing_labels.pickle",
        ssm1_output_pickle = "ssm1_output.pickle",
        # ── Training hyperparams ───────────────────────────────────
        num_epochs         = 20,
        learning_rate      = 1e-4,
        focal_gamma        = 2.0,
        class_weights      = None,   # torch.Tensor or None
        # ── SSM architecture ───────────────────────────────────────
        ssm_d_model        = 128,
        ssm_d_state        = 16,
        ssm_n_layers       = 2,
        # ── W&B ────────────────────────────────────────────────────
        wandb_run          = None,   # live wandb.Run object, or None
        wandb_log_freq     = 50,     # log every N training batches
        #── Checkpoint ───────────────────────────────────────────────
        checkpoint_path = ""
    ):
        self.data_dir             = Path(data_dir)
        self.preprocessing_script = preprocessing_script
        self.num_patches          = num_patches
        self.num_epochs           = num_epochs
        self.learning_rate        = learning_rate
        self.focal_gamma          = focal_gamma
        self.class_weights        = class_weights
        self.ssm_d_model          = ssm_d_model
        self.ssm_d_state          = ssm_d_state
        self.ssm_n_layers         = ssm_n_layers
        self.wandb_run            = wandb_run
        self.wandb_log_freq       = wandb_log_freq

        self.training_pickle    = self.data_dir / training_pickle
        self.testing_pickle     = self.data_dir / testing_pickle
        self.training_labels    = self.data_dir / training_labels
        self.testing_labels     = self.data_dir / testing_labels
        self.ssm1_output_pickle = self.data_dir / ssm1_output_pickle
        self.checkpoint_path    = self.data_dir / "checkpoint.pt"

    # ── Internal helpers ─────────────────────────────────────────────

    def _build_model(self):
        model = SeiSM(
            num_classes      = NUM_CLASSES,
            map_channels     = 5,
            catalog_features = 282,
            embed_dim        = EMBED_DIM,
            num_patches      = self.num_patches,
            d_model          = self.ssm_d_model,
            d_state          = self.ssm_d_state,
            n_ssm_layers     = self.ssm_n_layers,
        ).to(DEVICE)

        return model

    def _load_checkpoint(self):
        if not self.checkpoint_path.exists():
            raise RuntimeError(
                f"No checkpoint found at {self.checkpoint_path}. Run train() first."
            )
        model = self._build_model()
        checkpoint = torch.load(self.checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        return model

    def _forward(self, model, inputs):
        catalog = inputs["catalog"].to(DEVICE)
        maps    = inputs["maps"].to(DEVICE)

        logits = model({"catalog": catalog, "maps": maps})

        return logits, None

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
        print(f"Smoke test: SeiSM ({self.num_patches} patches)")
        print("=" * 60)

        model = self._build_model()

        batch      = 2
        fake_catalog = torch.randn(batch, SEQ_LEN, self.num_patches + 1, 282, device=DEVICE)
        fake_maps = torch.randn(batch, SEQ_LEN, self.num_patches, 50, 50, 5, device=DEVICE)

        out = model({"catalog": fake_catalog, "maps": fake_maps})

        print(f"  Catalog Input shape : {fake_catalog.shape}")
        print(f"  Maps Input shape    : {fake_maps.shape}")
        print(f"  Output shape        : {out.shape}  <- (batch, patches, num_classes)")
        print(f"  Device              : {DEVICE}")
        print(f"  Params              : {sum(p.numel() for p in model.parameters()):,}")
        print()

    def train(self, skip_preprocessing=False):
        """Train model end-to-end. Saves checkpoint and prints metrics."""
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
        print("Step 3: Building model...")
        print("=" * 60)
        model = self._build_model()
        print(f"  Model params     : {sum(p.numel() for p in model.parameters()):,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        weights   = self.class_weights.to(DEVICE) if self.class_weights is not None else None
        criterion = FocalLoss(alpha=weights, gamma=self.focal_gamma)

        if self.wandb_run is not None:
            self.wandb_run.watch(
                model,
                log="all",
                log_freq=self.wandb_log_freq,
            )

        print("\n" + "=" * 60)
        print(f"Step 4: Training for {self.num_epochs} epochs...")
        print("=" * 60)
        train_start  = time.time()
        global_step  = 0
        best_test_f1 = 0.0

        for epoch in range(self.num_epochs):
            model.train()

            epoch_loss    = 0.0
            epoch_correct = 0
            epoch_total   = 0

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                logits, _ = self._forward(model, inputs)
                loss = criterion(
                    logits.reshape(-1, NUM_CLASSES),
                    labels.reshape(-1),
                )
                loss.backward()
                optimizer.step()

                preds          = logits.argmax(dim=-1).reshape(-1)
                flat_labels    = labels.reshape(-1)
                batch_correct  = (preds == flat_labels).sum().item()
                batch_total    = flat_labels.numel()

                epoch_loss    += loss.item()
                epoch_correct += batch_correct
                epoch_total   += batch_total
                global_step   += 1

                if self.wandb_run is not None and global_step % self.wandb_log_freq == 0:
                    self.wandb_run.log(
                        {
                            "train/batch_loss":  loss.item(),
                            "train/batch_error": 1.0 - batch_correct / batch_total,
                            "train/lr":          optimizer.param_groups[0]["lr"],
                        },
                        step=global_step,
                    )

            avg_loss      = epoch_loss / len(train_loader)
            train_acc     = epoch_correct / epoch_total
            print(f"  Epoch {epoch+1}/{self.num_epochs}  loss: {avg_loss:.4f}  acc: {train_acc:.4f}")

            # ── Per-epoch evaluation ──────────────────────────────
            test_metrics = self._evaluate_split(model, test_loader)

            print(
                f"    Test  loss: {test_metrics['loss']:.4f}  "
                f"acc: {test_metrics['accuracy']:.4f}  "
                f"macro-F1: {test_metrics['f1_macro']:.4f}"
            )

            if self.wandb_run is not None:
                log_payload = {
                    "train/epoch_loss":     avg_loss,
                    "train/accuracy":       train_acc,
                    "train/error":          1.0 - train_acc,
                    "test/loss":            test_metrics["loss"],
                    "test/accuracy":        test_metrics["accuracy"],
                    "test/error":           test_metrics["error"],
                    "test/macro_f1":        test_metrics["f1_macro"],
                    "test/weighted_f1":     test_metrics["f1_weighted"],
                    "best/test_macro_f1":   best_test_f1,
                    "epoch":                epoch + 1,
                    "elapsed_time_sec":     time.time() - train_start, 
                }
                # Per-class metrics
                for i, name in enumerate(CLASS_NAMES):
                    log_payload[f"test/f1_{name}"]        = test_metrics["per_f1"][i]
                    log_payload[f"test/precision_{name}"]  = test_metrics["per_prec"][i]   
                    log_payload[f"test/recall_{name}"]     = test_metrics["per_rec"][i]
                    log_payload[f"test/support_{name}"]    = test_metrics["per_support"][i]

                self.wandb_run.log(log_payload, step=global_step)

            if test_metrics["f1_macro"] > best_test_f1:
                best_test_f1 = test_metrics["f1_macro"]
                torch.save({
                    "model": model.state_dict(),
                }, self.checkpoint_path)
                print(f"    *** New best checkpoint saved (F1: {best_test_f1:.4f}) → {self.checkpoint_path} ***")
                if self.wandb_run is not None:
                    self.wandb_run.summary["best_test_macro_f1"] = best_test_f1
                    self.wandb_run.save(str(self.checkpoint_path))

        training_time = time.time() - train_start
        print(f"\n  Training time: {training_time:.1f}s")

        print("\n" + "=" * 60)
        print("Final metrics on test set:")
        print("=" * 60)
        self.evaluate(model, test_loader)

    def _evaluate_split(self, model, loader) -> dict:
        """Run inference on *loader* and return a metrics dict."""
        model.eval()

        all_preds  = []
        all_labels = []
        total_loss = 0.0

        weights   = self.class_weights.to(DEVICE) if self.class_weights is not None else None
        criterion = FocalLoss(alpha=weights, gamma=self.focal_gamma)

        with torch.no_grad():
            for inputs, labels in loader:
                labels_dev = labels.to(DEVICE)
                logits, _  = self._forward(model, inputs)
                loss = criterion(
                    logits.reshape(-1, NUM_CLASSES),
                    labels_dev.reshape(-1),
                )
                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                all_preds.append(preds.cpu().reshape(-1))
                all_labels.append(labels.reshape(-1))

        all_preds  = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        per_prec, per_rec, per_f1, per_support = precision_recall_fscore_support(
            all_labels, all_preds, labels=list(range(NUM_CLASSES)), zero_division=0
        )
        accuracy = accuracy_score(all_labels, all_preds)

        return {
            "loss":        total_loss / len(loader),
            "accuracy":    accuracy,
            "error":       1.0 - accuracy,
            "f1_macro":    f1_score(all_labels, all_preds, average="macro",    zero_division=0),
            "f1_weighted": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
            "per_f1":      per_f1,
            "per_prec":    per_prec,       
            "per_rec":     per_rec,        
            "per_support": per_support,    
        }
    
    def evaluate(self, model, test_loader):
        metrics = self._evaluate_split(model, test_loader)

        print(f"  Accuracy         : {metrics['accuracy']:.4f}")
        print(f"  Macro F1         : {metrics['f1_macro']:.4f}")
        print(f"  Weighted F1      : {metrics['f1_weighted']:.4f}")
        print()
        print("  Per-class breakdown:")
        print(f"  {'Class':8s}  {'Prec':>8s}  {'Rec':>8s}  {'F1':>8s}  {'Support':>8s}")
        for i, name in enumerate(CLASS_NAMES):
            print(
                f"  {name:8s}  "
                f"{metrics['per_prec'][i]:>8.4f}  "
                f"{metrics['per_rec'][i]:>8.4f}  "
                f"{metrics['per_f1'][i]:>8.4f}  "
                f"{metrics['per_support'][i]:>8d}"
            )

    def run(self):
        """
        Load trained checkpoint and save SSM outputs to pickle for MLP fusion.
        Must call train() first.
        """
        print("=" * 60)
        print("Loading trained checkpoint...")
        print("=" * 60)
        model = self._load_checkpoint()
        model.eval()

        train_loader, test_loader = self._get_loaders(shuffle_train=False)

        def process_loader(loader, split_name):
            outputs = []
            labels  = []
            with torch.no_grad():
                for i, (inputs, label) in enumerate(loader):
                    logits, _ = self._forward(model, inputs)

                    outputs.append(logits.cpu())
                    labels.append(label)
                    if (i + 1) % 5 == 0:
                        print(f"  [{split_name}] {i+1}/{len(loader)} samples processed")
            return torch.cat(outputs, dim=0), torch.cat(labels, dim=0)

        print("\nGenerating model outputs (logits)...")
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
                "d_model":     self.ssm_d_model,
                "num_patches": self.num_patches,
                "num_classes": NUM_CLASSES,
            },
        }

        with open(self.ssm1_output_pickle, "wb") as f:
            pickle.dump(output, f)

        print(f"\n  Saved -> {self.ssm1_output_pickle}")
        print("Done.")