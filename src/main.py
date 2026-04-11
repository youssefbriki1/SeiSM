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
from utils import FocalLoss, SafeNetDataset
from models import QuakeMamba2
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "src" / "data-processing" / "california" / "data" / "CEED" / "processed"
DEFAULT_SAVE_PATH = PROJECT_ROOT / "checkpoints" / "best_quake_mamba2.pth"


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


def evaluate_split(model, dataloader, criterion, device, num_classes: int, desc: str):
    model.eval()
    split_loss = 0.0
    all_preds, all_targets = [], []
    split_bar = tqdm(dataloader, desc=desc)

    with torch.no_grad():
        for x, y in split_bar:
            x, y = x.to(device), y.to(device)

            logits = model(x)

            #trim the first element (the general patch)
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

    labels_list = list(range(num_classes))
    f1_per_class = f1_score(all_targets, all_preds, labels=labels_list, average=None, zero_division=0)
    precision_per_class = precision_score(all_targets, all_preds, labels=labels_list, average=None, zero_division=0)
    recall_per_class = recall_score(all_targets, all_preds, labels=labels_list, average=None, zero_division=0)

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "error": error,
        "f1": f1_score(all_targets, all_preds, average='macro', zero_division=0),
        "precision": precision_score(all_targets, all_preds, average='macro', zero_division=0),
        "recall": recall_score(all_targets, all_preds, average='macro', zero_division=0),
    }

    for i in range(num_classes):
        metrics[f"class_{i}_f1"] = float(f1_per_class[i])
        metrics[f"class_{i}_precision"] = float(precision_per_class[i])
        metrics[f"class_{i}_accuracy"] = float(recall_per_class[i])

    return metrics


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

    train_dataset = SafeNetDataset(
        train_features_path,
        train_labels_path if has_train_labels_file else None,
        labels_data=generated_train_labels,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = SafeNetDataset(
        val_features_path,
        val_labels_path if has_val_labels_file else None,
        labels_data=generated_val_labels,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = None
    if not args.skip_test_eval:
        test_dataset = SafeNetDataset(
            test_features_path,
            test_labels_path if has_test_labels_file else None,
            labels_data=generated_test_labels,
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Model, Loss, Optimizer ---
    sample_x, sample_y = train_dataset[0]
    _, num_patches, num_features = sample_x.shape
    model = QuakeMamba2(
        d_model=args.d_model,
        d_state=args.d_state,
        headdim=args.mamba_headdim,
        input_dim=num_patches * num_features,
        num_classes=args.num_classes,
        num_patches=num_patches,
        use_mem_eff_path=args.mamba_use_mem_eff_path,
    ).to(device)
    num_classes = args.num_classes

    if args.use_focal_loss:
        if args.focal_alpha is not None:
            if len(args.focal_alpha) != num_classes:
                raise ValueError(f"--focal_alpha must have {num_classes} elements, got {len(args.focal_alpha)}")
            class_weights = torch.tensor(args.focal_alpha, device=device)
        elif num_classes != 4:
            class_weights = torch.ones(num_classes, device=device)
        else:
            class_weights = torch.tensor([1.0, 4.0, 15.0, 78.0], device=device)
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
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
                "model": "QuakeMamba2",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "use_focal_loss": args.use_focal_loss,
                "focal_gamma": args.focal_gamma,
                "focal_alpha": args.focal_alpha,
                "data_dir": str(data_dir),
                "train_features_path": str(train_features_path),
                "train_labels_path": str(train_labels_path) if has_train_labels_file else None,
                "val_features_path": str(val_features_path),
                "val_labels_path": str(val_labels_path) if has_val_labels_file else None,
                "test_features_path": str(test_features_path) if not args.skip_test_eval else None,
                "test_labels_path": str(test_labels_path) if has_test_labels_file else None,
                "train_csv_path": str(train_csv_path),
                "val_csv_path": str(val_csv_path),
                "test_csv_path": str(test_csv_path) if not args.skip_test_eval else None,
                "patch_csv_path": str(patch_csv_path),
                "label_mag_bins": mag_bins,
                "save_path": str(save_path),
                "device": str(device),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "test_samples": len(test_loader.dataset) if test_loader is not None else 0,
                "skip_test_eval": args.skip_test_eval,
                "num_classes": num_classes,
                "num_patches": num_patches,
                "num_features": num_features,
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
            optimizer.zero_grad()
            
            for i, (x, y) in enumerate(train_bar):
                x = x.to(device)
                y = y.to(device)
                
                logits = model(x)

                # trim first patch (general map)
                num_label_patches = y.shape[-1]
                logits = logits[:, -num_label_patches:, :]
                logits_flat = logits.reshape(-1, num_classes)
                y_flat = y.reshape(-1)
                
                loss = criterion(logits_flat, y_flat)
                loss_item = loss.item()
                
                loss = loss / args.grad_accum_steps
                loss.backward()
                
                if (i + 1) % args.grad_accum_steps == 0 or (i + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                train_loss += loss_item
                preds = torch.argmax(logits_flat, dim=1)
                train_correct += (preds == y_flat).sum().item()
                train_total += y_flat.numel()
                global_step += 1
                train_bar.set_postfix({'loss': f"{loss_item:.4f}"})

                if wandb is not None:
                    wandb.log(
                        {
                            "train/batch_loss": loss_item,
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
            val_class_accs = " | ".join([f"C{i}: {val_metrics.get(f'class_{i}_accuracy', 0.0)*100:.1f}%" for i in range(num_classes)])
            val_class_precs = " | ".join([f"C{i}: {val_metrics.get(f'class_{i}_precision', 0.0):.4f}" for i in range(num_classes)])
            val_class_f1s = " | ".join([f"C{i}: {val_metrics.get(f'class_{i}_f1', 0.0):.4f}" for i in range(num_classes)])
            print(
                f"Val Macro-F1: {val_f1:.4f} | Precision: {val_precision:.4f} | "
                f"Recall: {val_recall:.4f}"
            )
            print(f"Val Class Accuracies: {val_class_accs}")
            print(f"Val Class Precisions: {val_class_precs}")
            print(f"Val Class F1:         {val_class_f1s}")
            if test_metrics is not None:
                test_class_accs = " | ".join([f"C{i}: {test_metrics.get(f'class_{i}_accuracy', 0.0)*100:.1f}%" for i in range(num_classes)])
                test_class_precs = " | ".join([f"C{i}: {test_metrics.get(f'class_{i}_precision', 0.0):.4f}" for i in range(num_classes)])
                test_class_f1s = " | ".join([f"C{i}: {test_metrics.get(f'class_{i}_f1', 0.0):.4f}" for i in range(num_classes)])
                print(
                    f"Test Loss: {test_metrics['loss']:.4f} | Test Error: {test_metrics['error']:.4f} | "
                    f"Test Macro-F1: {test_metrics['f1']:.4f}"
                )
                print(f"Test Class Accuracies: {test_class_accs}")
                print(f"Test Class Precisions: {test_class_precs}")
                print(f"Test Class F1:         {test_class_f1s}")

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
                for k, v in val_metrics.items():
                    if k.startswith("class_"):
                        log_payload[f"val/{k}"] = v
                """
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
                """
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
    parser = argparse.ArgumentParser(description="Train QuakeMamba2 for Earthquake Forecasting")
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing training/validation pickle files",
    )
    parser.add_argument("--train_features_file", type=str, default="ceed_training_output.pickle", help="Training features pickle file")
    parser.add_argument("--train_labels_file", type=str, default="ceed_training_labels.pickle", help="Training labels pickle file")
    parser.add_argument("--val_features_file", type=str, default="ceed_testing_output.pickle", help="Validation features pickle file")
    parser.add_argument("--val_labels_file", type=str, default="ceed_testing_labels.pickle", help="Validation labels pickle file")
    parser.add_argument("--test_features_file", type=str, default="ceed_testing_output.pickle", help="Test features pickle file")
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
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Number of steps to accumulate gradients before updating weights")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--d_model", type=int, default=128, help="Model hidden dimension")
    parser.add_argument("--d_state", type=int, default=16, help="Mamba state dimension")
    parser.add_argument("--mamba_headdim", type=int, default=32, help="Mamba head dimension (default 32 avoids stride issues)")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of classification classes")
    
    # Model configuration
    parser.add_argument("--mamba_use_mem_eff_path", action="store_true", help="Use Mamba's memory efficient path")
    parser.add_argument("--use_focal_loss", action="store_true", help="Flag to use Focal Loss instead of CrossEntropy")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma parameter for Focal Loss")
    parser.add_argument("--focal_alpha", type=float, nargs="+", default=None, help="Alpha class weights for Focal Loss, space separated (e.g., 1.0 4.0 15.0 78.0)")
    
    # Output arguments
    parser.add_argument("--save_path", type=str, default=str(DEFAULT_SAVE_PATH), help="Path to save the best model weights")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="quake-mamba2", help="W&B project name")
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
