import argparse
import os
import glob
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datasets import load_dataset

# Import the new Mamba2 model
from models import QuakeWaveMamba2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parent

# Paths to the cached Arrow directory and the downloaded CSV
DEFAULT_ARROW_PATH = "/scratch/brikiyou/ift3710/data/ceed_waveforms/AI4EPS___ceed/station_test/1.1.0/c062e7b0694b5aba3f4b3b624a764e52ecffbf5260ebfc550e1256de763c6e03"
DEFAULT_CSV_PATH = "/scratch/brikiyou/ift3710/data/ceed_waveforms/events_test.csv"
DEFAULT_SAVE_PATH = PROJECT_ROOT / "checkpoints" / "best_quake_mamba2_waveform.pth"

# ==========================================
# 1. ARROW WAVEFORM DATASET + CSV LABELS
# ==========================================
class ArrowSeismicDataset(torch.utils.data.Dataset):
    def __init__(self, arrow_dir_path: str, csv_path: str):
        if arrow_dir_path.endswith('.arrow'):
            arrow_dir_path = os.path.dirname(arrow_dir_path)
            
        print(f"Scanning directory for Arrow shards: {arrow_dir_path}")
        arrow_files = glob.glob(os.path.join(arrow_dir_path, "*.arrow"))
        arrow_files.sort()
        
        if not arrow_files:
            raise FileNotFoundError(f"Could not find any .arrow files in {arrow_dir_path}")
            
        print(f"Found {len(arrow_files)} Arrow shards. Stitching them together...")
        self.hf_dataset = load_dataset("arrow", data_files=arrow_files, split="train")
        
        print(f"Loading magnitude labels from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Create a lightning-fast O(1) lookup dictionary: {event_time: magnitude}
        # Fill missing magnitudes with 0.0 to prevent NaN poisoning
        self.mag_lookup = dict(zip(df['event_time'].astype(str), df['magnitude'].fillna(0.0).astype(float)))
        
        # Format for PyTorch but keep event_time accessible for the lookup
        self.hf_dataset = self.hf_dataset.with_format(type='torch', columns=['data'], output_all_columns=True)
        print(f"Successfully loaded {len(self.hf_dataset)} seismic events.")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        
        # 1. Extract the (3, 8192) waveform tensor
        waveform = sample['data']
        
        # 2. Scrub the raw data for dead sensors (NaNs or Infinities)
        waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 3. Lookup the magnitude using the timestamp
        event_time_str = str(sample['event_time'])
        mag_value = self.mag_lookup.get(event_time_str, 0.0)
        magnitude = torch.tensor([mag_value], dtype=torch.float32)
        
        # 4. Robust Z-score normalization per channel
        mean = waveform.mean(dim=1, keepdim=True)
        std = waveform.std(dim=1, keepdim=True) + 1e-5 # Higher epsilon to prevent div by zero
        waveform = (waveform - mean) / std
        
        # 5. Final scrub just in case normalization created new NaNs
        waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
            
        return waveform.float(), magnitude

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

def configure_hf_cache():
    if "HF_HOME" not in os.environ:
        hf_home = PROJECT_ROOT / ".cache" / "huggingface"
        hf_home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_home)

# ==========================================
# 2. EVALUATION FUNCTION (REGRESSION)
# ==========================================
def evaluate_split(model, dataloader, criterion, device, desc: str):
    model.eval()
    split_loss = 0.0
    all_preds, all_targets = [], []
    split_bar = tqdm(dataloader, desc=desc)

    with torch.no_grad():
        for x, y in split_bar:
            x, y = x.to(device), y.to(device)

            # FULL FP32 PRECISION (No Autocast)
            preds = model(x)
            loss = criterion(preds, y)
                
            split_loss += loss.item()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())

    avg_loss = split_loss / len(dataloader)
    
    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    return {
        "loss": avg_loss,
        "mae": mae,
        "mse": mse,
        "r2": r2,
    }

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train(args):
    configure_hf_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    arrow_path = str(resolve_path(args.arrow_path))
    csv_path = str(resolve_path(args.csv_path))
    
    full_dataset = ArrowSeismicDataset(arrow_path, csv_path)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    # Note: drop_last=True is critical for Mamba-2 to prevent weird batch sizes crashing causal_conv1d
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    # --- Model, Loss, Optimizer ---
    model = QuakeWaveMamba2(
        in_channels=3,
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        headdim=args.mamba_headdim,
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- W&B Setup ---
    wandb = None
    if not args.disable_wandb:
        try:
            import wandb as wandb_module
            wandb = wandb_module
            save_path = resolve_path(args.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity if args.wandb_entity else None,
                name=args.wandb_run_name if args.wandb_run_name else None,
                config=vars(args),
                mode=args.wandb_mode
            )
            wandb.watch(model, log="all")
        except ImportError:
            print("W&B not installed. Continuing without logging.")
    else:
        save_path = resolve_path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_mae = float('inf') 
    global_step = 0

    # --- Training Epochs ---
    try:
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            model.train()
            train_loss = 0.0
            train_bar = tqdm(train_loader, desc="Training")
            
            for x, y in train_bar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                
                # FULL FP32 PRECISION (No Autocast or Scaler)
                preds = model(x)
                loss = criterion(preds, y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                global_step += 1
                train_bar.set_postfix({'loss': f"{loss.item():.4f}"})

                if wandb is not None:
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch + 1,
                    }, step=global_step)

            avg_train_loss = train_loss / len(train_loader)
            
            val_metrics = evaluate_split(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                desc="Validation",
            )
            
            print(
                f"Train MAE Loss: {avg_train_loss:.4f} | "
                f"Val MAE Loss: {val_metrics['mae']:.4f} | Val R2: {val_metrics['r2']:.4f}"
            )

            if wandb is not None:
                wandb.log({
                    "train/epoch_loss": avg_train_loss,
                    "val/mae_loss": val_metrics["mae"],
                    "val/mse_loss": val_metrics["mse"],
                    "val/r2_score": val_metrics["r2"],
                    "best/val_mae": best_val_mae,
                    "epoch": epoch + 1,
                }, step=global_step)
            
            if val_metrics['mae'] < best_val_mae:
                best_val_mae = val_metrics['mae']
                torch.save(model.state_dict(), save_path)
                print(f"*** New best model saved to {save_path} with MAE: {best_val_mae:.4f} ***")

        print("Training Complete!")
    finally:
        if wandb is not None:
            wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QuakeWaveMamba2 for Seismic Regression")
    
    # Data arguments
    parser.add_argument("--arrow_path", type=str, default=DEFAULT_ARROW_PATH, help="Direct path to the cached .arrow dataset directory")
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH, help="Direct path to the events_test.csv file")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (Lowered for stability)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    
    # Mamba2 parameters
    parser.add_argument("--d_model", type=int, default=128, help="Model hidden dimension")
    parser.add_argument("--d_state", type=int, default=64, help="Mamba2 state dimension")
    parser.add_argument("--mamba_headdim", type=int, default=32, help="Mamba2 headdim. Set to 32 to ensure multiple-of-8 for causal_conv1d.")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of Mamba2 blocks")
    
    # Output arguments
    parser.add_argument("--save_path", type=str, default=str(DEFAULT_SAVE_PATH), help="Path to save the best model weights")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="quake-wave-mamba2", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default="", help="W&B entity/team")
    parser.add_argument("--wandb_run_name", type=str, default="run-1", help="W&B run name")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="offline",
        help="W&B mode: online, offline, or disabled",
    )

    args = parser.parse_args()
    
    train(args)