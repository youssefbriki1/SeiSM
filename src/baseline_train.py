"""
Baseline model training script.

Mirrors the structure of src/main.py (QuakeMamba2) but is model-agnostic.
Add new baselines to BASELINE_MODELS — each entry is a factory function that
receives (args, num_patches, num_features) and returns an nn.Module.

All models must share the same I/O contract as SafeNetDataset:
    input  : (batch, seq=10, num_patches=86, num_features=282)
    output : (batch, 85, num_classes)   — logits, patch 0 already dropped

Usage:
    python baseline_train.py --model lstm --epochs 50 --wandb_project baselines
    python baseline_train.py --model lstm --disable_wandb
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

SRC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_ROOT.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "src" / "data-processing" / "california" / "data" / "CEED" / "processed"
DEFAULT_CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

sys.path.insert(0, str(SRC_ROOT))
from utils import SafeNetDataset, MultimodalSafeNetDataset, FocalLoss


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
def _build_lstm(args, num_patches: int, num_features: int) -> nn.Module:
    from models.lstm.lstm import LSTMModel
    return LSTMModel(
        input_size=num_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        dropout=args.dropout,
    )


def _build_safenet_embeddings(args, num_patches: int, num_features: int) -> nn.Module:
    from models.safenet_embeddings import SafeNetEmbeddings
    return SafeNetEmbeddings(
        num_classes=args.num_classes,
        catalog_features=num_features,
        num_patches=num_patches - 1,  # 86 → 85 (global token excluded inside model)
    )


def _build_safenet_full(args, num_patches: int, num_features: int) -> nn.Module:
    from models.safenet_embeddings import SafeNetFull
    return SafeNetFull(
        num_classes=args.num_classes,
        catalog_features=num_features,
        num_patches=num_patches - 1,
        dropout=args.dropout,
    )


# To add a new baseline:
#   1. Write a _build_<name> factory above.
#   2. Register it here.
BASELINE_MODELS: dict[str, callable] = {
    "lstm": _build_lstm,
    "safenet_emb": _build_safenet_embeddings,
    "safenet_full": _build_safenet_full,
}

MULTIMODAL_MODELS = {"safenet_emb", "safenet_full"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def resolve_path(path_str: str, base_dir: Path | None = None) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    candidates = []
    if base_dir is not None:
        candidates.append(base_dir / path)
    candidates.extend([Path.cwd() / path, PROJECT_ROOT / path, SRC_ROOT / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (base_dir / path).resolve() if base_dir else (Path.cwd() / path).resolve()


def _to_device(x, device):
    """Move a tensor or a dict of tensors to *device*."""
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)


def evaluate_split(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    desc: str,
) -> dict:
    model.eval()
    split_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc=desc, leave=False):
            x, y = _to_device(x, device), y.to(device)
            logits = model(x).reshape(-1, num_classes)   # (batch*85, classes)
            y_flat = y.view(-1)                        # (batch*85,)

            split_loss += criterion(logits, y_flat).item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_targets.extend(y_flat.cpu().numpy())

    avg_loss = split_loss / len(dataloader)
    accuracy = sum(int(p == t) for p, t in zip(all_preds, all_targets)) / len(all_targets)
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "error": 1.0 - accuracy,
        "f1": f1_score(all_targets, all_preds, average="macro", zero_division=0),
        "precision": precision_score(all_targets, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_targets, all_preds, average="macro", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model: {args.model}")

    # ── Data ──────────────────────────────────────────────────────────────
    data_dir = resolve_path(args.data_dir)
    train_features = resolve_path(args.train_features_file, data_dir)
    train_labels   = resolve_path(args.train_labels_file,   data_dir)
    val_features   = resolve_path(args.val_features_file,   data_dir)
    val_labels     = resolve_path(args.val_labels_file,     data_dir)

    multimodal = args.model in MULTIMODAL_MODELS
    DatasetCls = MultimodalSafeNetDataset if multimodal else SafeNetDataset

    train_dataset = DatasetCls(train_features, train_labels)
    val_dataset   = DatasetCls(val_features,   val_labels)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

    test_loader = None
    if not args.skip_test_eval:
        test_features = resolve_path(args.test_features_file, data_dir)
        test_labels   = resolve_path(args.test_labels_file,   data_dir)
        test_dataset  = DatasetCls(test_features, test_labels)
        test_loader   = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    sample_x, _ = train_dataset[0]
    if isinstance(sample_x, dict):
        _, num_patches, num_features = sample_x["catalog"].shape  # (seq, patches, features)
    else:
        _, num_patches, num_features = sample_x.shape

    # ── Model ─────────────────────────────────────────────────────────────
    if args.model not in BASELINE_MODELS:
        raise ValueError(
            f"Unknown model '{args.model}'. Available: {list(BASELINE_MODELS)}"
        )
    model = BASELINE_MODELS[args.model](args, num_patches, num_features).to(device)
    num_classes = args.num_classes

    # ── Loss ──────────────────────────────────────────────────────────────
    if args.use_focal_loss:
        if args.focal_alpha is not None:
            if len(args.focal_alpha) != 4:
                raise ValueError(f"--focal_alpha must have 4 elements, got {len(args.focal_alpha)}")
            class_weights = torch.tensor(args.focal_alpha, device=device)
        else:
            class_weights = torch.tensor([1.0, 4.0, 15.0, 78.0], device=device)
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
        print("Using Focal Loss to handle class imbalance.")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using Cross Entropy Loss.")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ── W&B ───────────────────────────────────────────────────────────────
    wandb = None
    save_path = DEFAULT_CHECKPOINTS_DIR / f"best_{args.model}.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.disable_wandb:
        try:
            import wandb as _wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Run `pip install wandb` or pass --disable_wandb."
            ) from exc
        wandb = _wandb
        wandb.init(
            # project=args.wandb_project,
            project=f"{args.model}_AdamW_lr({args.lr})_bs({args.batch_size})",
            entity="ift3710-ai-slop",
            name=args.wandb_run_name or f"{args.model}-baseline",
            mode=args.wandb_mode,
            config={
                "model": args.model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "num_classes": num_classes,
                "num_patches": num_patches,
                "num_features": num_features,
                "use_focal_loss": args.use_focal_loss,
                "focal_gamma": args.focal_gamma,
                "focal_alpha": args.focal_alpha,
                "data_dir": str(data_dir),
            },
        )
        wandb.watch(model, log="all", log_freq=args.wandb_log_freq)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_f1 = 0.0
    global_step = 0

    try:
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for x, y in tqdm(train_loader, desc="Training"):
                x, y = _to_device(x, device), y.to(device)
                optimizer.zero_grad()

                logits = model(x).reshape(-1, num_classes)   # (batch*85, classes)
                y_flat = y.view(-1)                        # (batch*85,)

                loss = criterion(logits, y_flat)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                preds = torch.argmax(logits, dim=1)
                train_loss += loss.item()
                train_correct += (preds == y_flat).sum().item()
                train_total += y_flat.numel()
                global_step += 1

                if wandb is not None:
                    wandb.log(
                        {
                            "train/batch_loss": loss.item(),
                            "train/batch_error": 1.0 - (preds == y_flat).float().mean().item(),
                            "train/lr": optimizer.param_groups[0]["lr"],
                        },
                        step=global_step,
                    )

            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total

            val_metrics = evaluate_split(model, val_loader, criterion, device, num_classes, "Validation")
            test_metrics = None
            if test_loader is not None:
                test_metrics = evaluate_split(model, test_loader, criterion, device, num_classes, "Test")

            print(
                f"Train Loss: {avg_train_loss:.4f} | Train Error: {1 - train_accuracy:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | Val Error: {val_metrics['error']:.4f}"
            )
            print(
                f"Val Macro-F1: {val_metrics['f1']:.4f} | "
                f"Precision: {val_metrics['precision']:.4f} | "
                f"Recall: {val_metrics['recall']:.4f}"
            )
            if test_metrics is not None:
                print(
                    f"Test Loss: {test_metrics['loss']:.4f} | "
                    f"Test Error: {test_metrics['error']:.4f} | "
                    f"Test Macro-F1: {test_metrics['f1']:.4f}"
                )

            if wandb is not None:
                log_payload = {
                    "train/epoch_loss": avg_train_loss,
                    "train/accuracy": train_accuracy,
                    "train/error": 1.0 - train_accuracy,
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/error": val_metrics["error"],
                    "val/macro_f1": val_metrics["f1"],
                    "val/precision": val_metrics["precision"],
                    "val/recall": val_metrics["recall"],
                    "best/val_macro_f1": best_val_f1,
                    "epoch": epoch + 1,
                }
                if test_metrics is not None:
                    log_payload.update({
                        "test/loss": test_metrics["loss"],
                        "test/accuracy": test_metrics["accuracy"],
                        "test/error": test_metrics["error"],
                        "test/macro_f1": test_metrics["f1"],
                        "test/precision": test_metrics["precision"],
                        "test/recall": test_metrics["recall"],
                    })
                wandb.log(log_payload, step=global_step)

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                torch.save(model.state_dict(), save_path)
                print(f"*** New best model saved → {save_path}  (F1: {best_val_f1:.4f}) ***")
                if wandb is not None:
                    wandb.run.summary["best_val_macro_f1"] = best_val_f1
                    wandb.save(str(save_path))

        print("\nTraining complete!")

    finally:
        if wandb is not None:
            wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a baseline model for earthquake forecasting")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        choices=list(BASELINE_MODELS),
        help="Baseline model to train",
    )

    # Data
    parser.add_argument("--data_dir",           type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--train_features_file", type=str, default="ceed_training_output.pickle")
    parser.add_argument("--train_labels_file",   type=str, default="ceed_training_labels.pickle")
    parser.add_argument("--val_features_file",   type=str, default="ceed_testing_output.pickle")
    parser.add_argument("--val_labels_file",     type=str, default="ceed_testing_labels.pickle")
    parser.add_argument("--test_features_file",  type=str, default="ceed_testing_output.pickle")
    parser.add_argument("--test_labels_file",    type=str, default="ceed_testing_labels.pickle")
    parser.add_argument("--skip_test_eval", action="store_true")

    # Training hyper-parameters
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_classes",  type=int,   default=4)

    # LSTM-specific hyper-parameters
    parser.add_argument("--hidden_size", type=int,   default=128)
    parser.add_argument("--num_layers",  type=int,   default=2)
    parser.add_argument("--dropout",     type=float, default=0.3)

    # Loss configuration
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, nargs="+", default=None)

    # W&B
    parser.add_argument("--disable_wandb",  action="store_true")
    parser.add_argument("--wandb_project",  type=str, default="baselines")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--wandb_log_freq", type=int, default=100)

    train(parser.parse_args())
