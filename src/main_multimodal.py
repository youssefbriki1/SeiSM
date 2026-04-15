import argparse
import os
import pickle
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import pandas as pd
from utils import FocalLoss, MultimodalSafeNetDataset
from models.safenet_embeddings import SafeNetFull

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "src" / "data-processing" / "california" / "data" / "CEED" / "processed"
DEFAULT_SAVE_PATH = PROJECT_ROOT / "checkpoints" / "best_safenet_full.pth"


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

    if base_dir is not None:
        return (base_dir / path).resolve()
    return (Path.cwd() / path).resolve()


def is_git_lfs_pointer(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False

    with path.open("rb") as f:
        header = f.read(64)
    return header.startswith(b"version https://git-lfs.github.com/spec/v1")


def ensure_readable_file(path: Path, label: str, required: bool = True) -> bool:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing {label}: {path}")
        return False
    if is_git_lfs_pointer(path):
        raise RuntimeError(
            f"{label} is a Git LFS pointer, not the real file: {path}\n"
            "Run `git lfs pull` to fetch actual dataset files."
        )
    return True


def configure_hf_cache():
    transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
    if transformers_cache and not os.access(transformers_cache, os.W_OK):
        os.environ.pop("TRANSFORMERS_CACHE", None)

    if "HF_HOME" not in os.environ:
        hf_home = PROJECT_ROOT / ".cache" / "huggingface"
        hf_home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_home)


def parse_mag_bins(bins_str: str) -> list[float]:
    return [float(x.strip()) for x in bins_str.split(",") if x.strip()]


def magnitude_to_class(max_magnitude: float, thresholds: list[float]) -> int:
    class_idx = 0
    for threshold in thresholds:
        if max_magnitude >= threshold:
            class_idx += 1
    return class_idx


def feature_metadata(features_path: Path) -> tuple[int, int]:
    with features_path.open("rb") as f:
        features_obj = pickle.load(f)

    if isinstance(features_obj, dict):
        eq_data = features_obj["eq_data"]
    else:
        eq_data = features_obj

    num_samples = len(eq_data)
    num_patches = eq_data[0].shape[1]
    return num_samples, num_patches


def build_labels_from_csv(
    csv_path: Path,
    patch_csv_path: Path,
    target_year_start: int,
    num_samples: int,
    num_patches: int,
    mag_bins: list[float],
):
    df = pd.read_csv(csv_path)
    if "onlydate" not in df.columns or "magnitude" not in df.columns:
        raise ValueError(f"CSV is missing required columns: {csv_path}")

    df["year"] = pd.to_datetime(df["onlydate"], errors="coerce").dt.year
    df = df.dropna(subset=["year", "magnitude"])
    df["year"] = df["year"].astype(int)

    if "need" in df.columns:
        need_series = df["need"].astype(str).str.lower()
        df = df[need_series.isin({"true", "1", "yes"})]

    patch_df = pd.read_csv(patch_csv_path)
    patch_regions = [f"({int(row.x)}, {int(row.y)})" for row in patch_df.itertuples(index=False)]
    patch_regions = patch_regions[: max(0, num_patches - 1)]

    year_max = df.groupby("year")["magnitude"].max().to_dict()
    year_region_max = df.groupby(["year", "region"])["magnitude"].max().to_dict()

    labels = []
    for target_year in range(target_year_start, target_year_start + num_samples):
        general_mag = float(year_max.get(target_year, float("-inf")))
        row = [magnitude_to_class(general_mag, mag_bins)]

        for region in patch_regions:
            region_mag = float(year_region_max.get((target_year, region), float("-inf")))
            row.append(magnitude_to_class(region_mag, mag_bins))

        if len(row) < num_patches:
            row.extend([0] * (num_patches - len(row)))
        labels.append(row[:num_patches])

    return labels


def evaluate_split(model, dataloader, criterion, device, num_classes: int, num_patches: int, desc: str):
    """Evaluate model on a split. Handles multimodal dict inputs."""
    model.eval()
    split_loss = 0.0
    all_preds, all_targets = [], []
    split_bar = tqdm(dataloader, desc=desc)

    with torch.no_grad():
        for inputs, y in split_bar:
            # Move dict inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            y = y.to(device)

            logits = model(inputs)  # (B, num_patches, num_classes)

            # Trim to match label patches (labels are for regional patches only)
            num_label_patches = y.shape[-1]
            logits = logits[:, -num_label_patches:, :]
            logits_flat = logits.reshape(-1, num_classes)
            y_flat = y.reshape(-1)

            loss = criterion(logits_flat, y_flat)
            split_loss += loss.item()

            preds = torch.argmax(logits_flat, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_flat.cpu().numpy())

    avg_loss = split_loss / len(dataloader)
    accuracy = sum(int(p == t) for p, t in zip(all_preds, all_targets)) / len(all_targets)
    error = 1.0 - accuracy

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "error": error,
        "f1": f1_score(all_targets, all_preds, average='macro', zero_division=0),
        "precision": precision_score(all_targets, all_preds, average='macro', zero_division=0),
        "recall": recall_score(all_targets, all_preds, average='macro', zero_division=0),
    }


def train(args):
    configure_hf_cache()

    # --- Setup & Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    data_dir = resolve_path(args.data_dir)
    train_features_path = resolve_path(args.train_features_file, base_dir=data_dir)
    train_labels_path = resolve_path(args.train_labels_file, base_dir=data_dir)
    val_features_path = resolve_path(args.val_features_file, base_dir=data_dir)
    val_labels_path = resolve_path(args.val_labels_file, base_dir=data_dir)
    test_features_path = resolve_path(args.test_features_file, base_dir=data_dir)
    test_labels_path = resolve_path(args.test_labels_file, base_dir=data_dir)
    train_csv_path = resolve_path(args.train_csv_file, base_dir=data_dir)
    val_csv_path = resolve_path(args.val_csv_file, base_dir=data_dir)
    test_csv_path = resolve_path(args.test_csv_file, base_dir=data_dir)
    patch_csv_path = resolve_path(args.patch_csv_file, base_dir=data_dir)
    mag_bins = parse_mag_bins(args.label_mag_bins)

    ensure_readable_file(train_features_path, "training features file")
    ensure_readable_file(val_features_path, "validation features file")
    has_train_labels_file = ensure_readable_file(train_labels_path, "training labels file", required=False)
    has_val_labels_file = ensure_readable_file(val_labels_path, "validation labels file", required=False)
    has_test_labels_file = False
    if not args.skip_test_eval:
        ensure_readable_file(test_features_path, "test features file")
        has_test_labels_file = ensure_readable_file(test_labels_path, "test labels file", required=False)

    csv_labels_enabled = not args.disable_csv_label_fallback
    generated_train_labels = None
    generated_val_labels = None
    generated_test_labels = None
    if csv_labels_enabled and (not has_train_labels_file or not has_val_labels_file or (not args.skip_test_eval and not has_test_labels_file)):
        ensure_readable_file(train_csv_path, "training CSV file")
        ensure_readable_file(val_csv_path, "validation CSV file")
        if not args.skip_test_eval:
            ensure_readable_file(test_csv_path, "test CSV file")
        ensure_readable_file(patch_csv_path, "patch mapping CSV file")

    if not has_train_labels_file:
        print(
            f"[Data] training labels file not found at {train_labels_path}. "
            "Will attempt labels from the features pickle."
        )
    if not has_val_labels_file:
        print(
            f"[Data] validation labels file not found at {val_labels_path}. "
            "Will attempt labels from the features pickle."
        )
    if not args.skip_test_eval and not has_test_labels_file:
        print(
            f"[Data] test labels file not found at {test_labels_path}. "
            "Will attempt labels from the features pickle."
        )

    if csv_labels_enabled and not has_train_labels_file:
        n_train, p_train = feature_metadata(train_features_path)
        generated_train_labels = build_labels_from_csv(
            csv_path=train_csv_path,
            patch_csv_path=patch_csv_path,
            target_year_start=args.train_target_year_start,
            num_samples=n_train,
            num_patches=p_train,
            mag_bins=mag_bins,
        )
        print(f"[Data] Generated {len(generated_train_labels)} training labels from {train_csv_path}.")
    if csv_labels_enabled and not has_val_labels_file:
        n_val, p_val = feature_metadata(val_features_path)
        generated_val_labels = build_labels_from_csv(
            csv_path=val_csv_path,
            patch_csv_path=patch_csv_path,
            target_year_start=args.val_target_year_start,
            num_samples=n_val,
            num_patches=p_val,
            mag_bins=mag_bins,
        )
        print(f"[Data] Generated {len(generated_val_labels)} validation labels from {val_csv_path}.")
    if csv_labels_enabled and not args.skip_test_eval and not has_test_labels_file:
        n_test, p_test = feature_metadata(test_features_path)
        generated_test_labels = build_labels_from_csv(
            csv_path=test_csv_path,
            patch_csv_path=patch_csv_path,
            target_year_start=args.test_target_year_start,
            num_samples=n_test,
            num_patches=p_test,
            mag_bins=mag_bins,
        )
        print(f"[Data] Generated {len(generated_test_labels)} test labels from {test_csv_path}.")

    # --- Multimodal datasets (catalog + maps) ---
    train_dataset = MultimodalSafeNetDataset(
        train_features_path,
        train_labels_path if has_train_labels_file else None,
        labels_data=generated_train_labels,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = MultimodalSafeNetDataset(
        val_features_path,
        val_labels_path if has_val_labels_file else None,
        labels_data=generated_val_labels,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = None
    if not args.skip_test_eval:
        test_dataset = MultimodalSafeNetDataset(
            test_features_path,
            test_labels_path if has_test_labels_file else None,
            labels_data=generated_test_labels,
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Inspect sample to determine dimensions ---
    sample_inputs, sample_y = train_dataset[0]
    catalog_shape = sample_inputs["catalog"].shape   # (T=10, P+1, 282)
    maps_shape = sample_inputs["maps"].shape         # (T=10, P, H, W, C)
    num_patches = maps_shape[1]                      # P = regional patches (e.g. 64)
    catalog_features = catalog_shape[-1]             # 282
    map_channels = maps_shape[-1]                    # 5
    print(f"[Data] catalog shape: {catalog_shape}, maps shape: {maps_shape}")
    print(f"[Data] num_patches={num_patches}, catalog_features={catalog_features}, map_channels={map_channels}")

    # --- Model: SafeNetFull (multimodal: catalog + maps → LSTM → ViT → classification) ---
    num_classes = args.num_classes
    model = SafeNetFull(
        num_classes=num_classes,
        map_channels=map_channels,
        catalog_features=catalog_features,
        embed_dim=args.embed_dim,
        num_patches=num_patches,
        num_heads=args.num_heads,
        transformer_layers=args.transformer_layers,
        dropout=args.dropout,
    ).to(device)
    print(f"[Model] SafeNetFull — embed_dim={args.embed_dim}, num_heads={args.num_heads}, "
          f"transformer_layers={args.transformer_layers}, dropout={args.dropout}")

    if args.use_focal_loss:
        if num_classes != 4:
            class_weights = torch.ones(num_classes, device=device)
        else:
            class_weights = torch.tensor([0.1, 1.0, 5.0, 20.0], device=device)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        print("Using Focal Loss to handle class imbalance.")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard Cross Entropy Loss.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    wandb = None
    if not args.disable_wandb:
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Install it with `python -m pip install wandb`, "
                "or pass --disable_wandb."
            ) from exc

        wandb = wandb_module
        save_path = resolve_path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity if args.wandb_entity else None,
            name=args.wandb_run_name if args.wandb_run_name else None,
            mode=args.wandb_mode,
            config={
                "model": "SafeNetFull",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "embed_dim": args.embed_dim,
                "num_heads": args.num_heads,
                "transformer_layers": args.transformer_layers,
                "dropout": args.dropout,
                "use_focal_loss": args.use_focal_loss,
                "data_dir": str(data_dir),
                "train_features_path": str(train_features_path),
                "train_labels_path": str(train_labels_path) if has_train_labels_file else None,
                "val_features_path": str(val_features_path),
                "val_labels_path": str(val_labels_path) if has_val_labels_file else None,
                "test_features_path": str(test_features_path) if not args.skip_test_eval else None,
                "test_labels_path": str(test_labels_path) if has_test_labels_file else None,
                "label_mag_bins": mag_bins,
                "save_path": str(save_path),
                "device": str(device),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "test_samples": len(test_loader.dataset) if test_loader is not None else 0,
                "skip_test_eval": args.skip_test_eval,
                "num_classes": num_classes,
                "num_patches": num_patches,
                "catalog_features": catalog_features,
                "map_channels": map_channels,
            },
        )
        wandb.watch(model, log="all", log_freq=args.wandb_log_freq)
    else:
        save_path = resolve_path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0 
    global_step = 0

    # --- Training Loop ---
    try:
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # 1. Training Phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_bar = tqdm(train_loader, desc="Training")
            
            for inputs, y in train_bar:
                # Move dict inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                y = y.to(device)
                optimizer.zero_grad()
                
                logits = model(inputs)  # (B, num_patches, num_classes)

                # Trim to match label patches (labels are for regional patches only)
                num_label_patches = y.shape[-1]
                logits = logits[:, -num_label_patches:, :]
                logits_flat = logits.reshape(-1, num_classes)
                y_flat = y.reshape(-1)
                
                loss = criterion(logits_flat, y_flat)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                preds = torch.argmax(logits_flat, dim=1)
                train_correct += (preds == y_flat).sum().item()
                train_total += y_flat.numel()
                global_step += 1
                train_bar.set_postfix({'loss': f"{loss.item():.4f}"})

                if wandb is not None:
                    wandb.log(
                        {
                            "train/batch_loss": loss.item(),
                            "train/batch_error": 1.0 - ((preds == y_flat).float().mean().item()),
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/epoch": epoch + 1,
                        },
                        step=global_step,
                    )

            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            train_error = 1.0 - train_accuracy
            
            # 2. Validation Phase
            val_metrics = evaluate_split(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=num_classes,
                num_patches=num_patches,
                desc="Validation",
            )
            test_metrics = None
            if test_loader is not None:
                test_metrics = evaluate_split(
                    model=model,
                    dataloader=test_loader,
                    criterion=criterion,
                    device=device,
                    num_classes=num_classes,
                    num_patches=num_patches,
                    desc="Test",
                )
            
            # 3. Metrics & Saving
            val_f1 = val_metrics["f1"]
            val_precision = val_metrics["precision"]
            val_recall = val_metrics["recall"]
            
            print(
                f"Train Loss: {avg_train_loss:.4f} | Train Error: {train_error:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | Val Error: {val_metrics['error']:.4f}"
            )
            print(
                f"Val Macro-F1: {val_f1:.4f} | Precision: {val_precision:.4f} | "
                f"Recall: {val_recall:.4f}"
            )
            if test_metrics is not None:
                print(
                    f"Test Loss: {test_metrics['loss']:.4f} | Test Error: {test_metrics['error']:.4f} | "
                    f"Test Macro-F1: {test_metrics['f1']:.4f}"
                )

            if wandb is not None:
                log_payload = {
                    "train/epoch_loss": avg_train_loss,
                    "train/accuracy": train_accuracy,
                    "train/error": train_error,
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/error": val_metrics["error"],
                    "val/macro_f1": val_f1,
                    "val/precision": val_precision,
                    "val/recall": val_recall,
                    "best/val_macro_f1": best_val_f1,
                    "epoch": epoch + 1,
                }
                if test_metrics is not None:
                    log_payload.update(
                        {
                            "test/loss": test_metrics["loss"],
                            "test/accuracy": test_metrics["accuracy"],
                            "test/error": test_metrics["error"],
                            "test/macro_f1": test_metrics["f1"],
                            "test/precision": test_metrics["precision"],
                            "test/recall": test_metrics["recall"],
                        }
                    )
                wandb.log(log_payload, step=global_step)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), save_path)
                print(f"*** New best model saved to {save_path} with F1: {best_val_f1:.4f} ***")

                if wandb is not None:
                    wandb.run.summary["best_val_macro_f1"] = best_val_f1
                    wandb.save(str(save_path))

        print("Training Complete!")
    finally:
        if wandb is not None:
            wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SafeNetFull (Multimodal) for Earthquake Forecasting")
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing training/validation pickle files",
    )
    parser.add_argument("--train_features_file", type=str, default="ceed_training_output.pickle", help="Training features pickle file (must contain eq_data + png)")
    parser.add_argument("--train_labels_file", type=str, default="ceed_training_labels.pickle", help="Training labels pickle file")
    parser.add_argument("--val_features_file", type=str, default="ceed_testing_output.pickle", help="Validation features pickle file (must contain eq_data + png)")
    parser.add_argument("--val_labels_file", type=str, default="ceed_testing_labels.pickle", help="Validation labels pickle file")
    parser.add_argument("--test_features_file", type=str, default="ceed_testing_output.pickle", help="Test features pickle file (must contain eq_data + png)")
    parser.add_argument("--test_labels_file", type=str, default="ceed_testing_labels.pickle", help="Test labels pickle file")
    parser.add_argument("--skip_test_eval", action="store_true", help="Disable test evaluation during training")
    parser.add_argument("--train_csv_file", type=str, default="training_data.csv", help="Training CSV source for label generation")
    parser.add_argument("--val_csv_file", type=str, default="testing_data.csv", help="Validation CSV source for label generation")
    parser.add_argument("--test_csv_file", type=str, default="testing_data.csv", help="Test CSV source for label generation")
    parser.add_argument("--patch_csv_file", type=str, default="png_list_to_patchxy.csv", help="Patch mapping CSV (x,y order)")
    parser.add_argument("--disable_csv_label_fallback", action="store_true", help="Disable CSV-based label generation fallback")
    parser.add_argument("--label_mag_bins", type=str, default="5,6,7", help="Comma-separated magnitude bins for class labels")
    parser.add_argument("--train_target_year_start", type=int, default=1970, help="Target start year for training labels")
    parser.add_argument("--val_target_year_start", type=int, default=2011, help="Target start year for validation labels")
    parser.add_argument("--test_target_year_start", type=int, default=2011, help="Target start year for test labels")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (smaller default due to map memory usage)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of classification classes")

    # SafeNetFull architecture hyperparameters
    parser.add_argument("--embed_dim", type=int, default=32, help="Embedding dimension for both catalog and map encoders")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads in the Vision Transformer")
    parser.add_argument("--transformer_layers", type=int, default=1, help="Number of Transformer encoder layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate in the Transformer encoder")
    
    # Model/Loss configuration
    parser.add_argument("--use_focal_loss", action="store_true", help="Flag to use Focal Loss instead of CrossEntropy")
    
    # Output arguments
    parser.add_argument("--save_path", type=str, default=str(DEFAULT_SAVE_PATH), help="Path to save the best model weights")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="safenet-full", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default="", help="W&B entity/team (optional)")
    parser.add_argument("--wandb_run_name", type=str, default="", help="W&B run name (optional)")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
        help="W&B mode: online, offline, or disabled",
    )
    parser.add_argument("--wandb_log_freq", type=int, default=100, help="Frequency for gradient/parameter logging")
    
    args = parser.parse_args()
    
    train(args)