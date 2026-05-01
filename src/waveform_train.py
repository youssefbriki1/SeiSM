import argparse
import os
import glob
import json
import random
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datasets import load_dataset
import torch.optim.lr_scheduler as lr_scheduler
from models import QuakeWaveMamba2, BiWaveformLSTM, WaveformTransformer
import warnings
from utils import Muon, BinnedWeightedMSELoss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parent

DEFAULT_ARROW_PATHS = [
    "/scratch/brikiyou/ift3710/data/ceed_waveforms/AI4EPS___ceed/station_test/1.1.0/c062e7b0694b5aba3f4b3b624a764e52ecffbf5260ebfc550e1256de763c6e03",
    #"/scratch/brikiyou/ift3710/data/ceed_waveforms/AI4EPS___ceed/station_test/1.1.0/augmented_data"
]
DEFAULT_CSV_PATH = "/scratch/brikiyou/ift3710/data/ceed_waveforms/events_test.csv"
DEFAULT_SAVE_PATH = PROJECT_ROOT / "checkpoints" / "best_quake_mamba2_waveform.pth"

class ArrowSeismicDataset(torch.utils.data.Dataset):
    def __init__(self, arrow_dir_path: str, csv_path: str):
        if arrow_dir_path.endswith('.arrow'):
            arrow_dir_path = os.path.dirname(arrow_dir_path)
            
        print(f"Scanning directory for Arrow shards: {arrow_dir_path}")
        arrow_files = glob.glob(os.path.join(arrow_dir_path, "*.arrow"))
        arrow_files.sort()
        
        if not arrow_files:
            raise FileNotFoundError(f"Could not find any .arrow files in {arrow_dir_path}")
            
        self.hf_dataset = load_dataset("arrow", data_files=arrow_files, split="train")
        
        df = pd.read_csv(csv_path)
        self.mag_lookup = dict(zip(df['event_time'].astype(str), df['magnitude'].fillna(0.0).astype(float)))
        
        self.hf_dataset = self.hf_dataset.with_format(type='torch', columns=['data'], output_all_columns=True)
        print(f"Successfully loaded {len(self.hf_dataset)} events from {arrow_dir_path}")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        
        waveform = sample['data']
        waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        
        event_time_str = str(sample['event_time'])
        mag_value = self.mag_lookup.get(event_time_str, 0.0)
        magnitude = torch.tensor([mag_value], dtype=torch.float32)
        
        mean = waveform.mean(dim=1, keepdim=True)
        std = waveform.std(dim=1, keepdim=True) + 1e-5
        waveform = (waveform - mean) / std
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
# 2. EVALUATION FUNCTION
# ==========================================
def evaluate_split(model, dataloader, criterion, device, desc: str):
    model.eval()
    split_loss = 0.0
    all_preds, all_targets = [], []
    split_bar = tqdm(dataloader, desc=desc)

    with torch.no_grad():
        for x, y in split_bar:
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                preds = model(x)
                loss = criterion(preds, y)
                
            split_loss += loss.item()

            all_preds.extend(preds.float().cpu().numpy().flatten())
            all_targets.extend(y.float().cpu().numpy().flatten())

    avg_loss = split_loss / len(dataloader)
    
    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    return {
        "loss": avg_loss,
        "mae": float(mae),
        "mse": float(mse),
        "r2": float(r2),
    }

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train(args):
    configure_hf_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading (PyTorch Concat Bypass) ---
    csv_path = str(resolve_path(args.csv_path))
    
    individual_datasets = []
    for path in args.arrow_paths:
        ds = ArrowSeismicDataset(str(resolve_path(path)), csv_path)
        individual_datasets.append(ds)
        
    print("Stitching datasets together via PyTorch ConcatDataset...")
    full_dataset = ConcatDataset(individual_datasets)
    print(f"Total Combined Dataset Size: {len(full_dataset)}")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Split: {train_size} train / {val_size} val (seed={args.seed})")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    # --- Model, Loss, Optimizer ---
    if args.model_type == "mamba2":
        model = QuakeWaveMamba2(
        ).to(device)
    elif args.model_type == "bi_lstm":
        model = BiWaveformLSTM(
        ).to(device)
    elif args.model_type == "transformer":
        model = WaveformTransformer(
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    #print("Compiling model for Hopper Tensor Cores...")
    #model = torch.compile(model)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Training model...")
    
    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "l1":
        criterion = nn.L1Loss()
    elif args.loss == "binned_mse":
        bin_edges = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        bin_counts = [16690, 29767, 8355, 1491, 183, 24]
        criterion = BinnedWeightedMSELoss(bin_edges=bin_edges, bin_counts=bin_counts)
    else:
        warnings.warn(f"Unknown loss type: {args.loss}. Defaulting to MSELoss.")
        criterion = nn.MSELoss()

    criterion.to(device)
    
    
    optimizers = []
    
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizers.append(optimizer)
        
    elif args.optimizer == "muon":
        muon_params = []
        adamw_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 2:
                muon_params.append(param)
            else:
                adamw_params.append(param)
                
        opt_muon = Muon(muon_params, lr=args.lr, weight_decay=args.weight_decay, adjust_lr_fn="match_rms_adamw")
        opt_adamw = torch.optim.AdamW(adamw_params, lr=args.lr, weight_decay=args.weight_decay)
        
        optimizers.extend([opt_muon, opt_adamw])
        
    else:
        warnings.warn(f"Unknown optimizer type: {args.optimizer}. Defaulting to AdamW.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizers.append(optimizer)

    schedulers = [] 
    if args.lr_scheduler == "cosine":
        steps_per_epoch = len(train_loader)
        warmup_steps = args.warmup_epochs * steps_per_epoch
        total_steps = args.epochs * steps_per_epoch
        
        for opt in optimizers:
            cosine_scheduler = lr_scheduler.CosineAnnealingLR(
                opt, 
                T_max=(total_steps - warmup_steps)
            )
            
            warmup_scheduler = lr_scheduler.LinearLR(
                opt, 
                start_factor=0.01, 
                total_iters=max(1, warmup_steps)
            )
            
            scheduler = lr_scheduler.SequentialLR(
                opt,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps] 
            )
            schedulers.append(scheduler)            
            
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
        except ImportError:
            print("W&B not installed. Continuing without logging.")
    else:
        save_path = resolve_path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_mae = float('inf')
    global_step = 0
    global_start_time = time.time()
    epoch_history = []

    # --- Training Epochs ---
    try:
        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            model.train()
            train_loss = 0.0
            train_bar = tqdm(train_loader, desc="Training")
            
            for x, y in train_bar:
                x, y = x.to(device), y.to(device)
                
                for opt in optimizers:
                    opt.zero_grad()
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    preds = model(x)
                    loss = criterion(preds, y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                for opt in optimizers:
                    opt.step()
                
                train_loss += loss.item()
                global_step += 1
                train_bar.set_postfix({'loss': f"{loss.item():.4f}"})

                if wandb is not None:
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/lr": optimizers[0].param_groups[0]["lr"],
                        "train/epoch": epoch + 1,
                    }, step=global_step)
            
                for sched in schedulers:
                    sched.step()
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

            epoch_time = time.time() - epoch_start_time

            if wandb is not None:
                wandb.log({
                    "train/epoch_loss": avg_train_loss,
                    "val/mae_loss": val_metrics["mae"],
                    "val/mse_loss": val_metrics["mse"],
                    "val/r2_score": val_metrics["r2"],
                    "time/epoch_time_seconds": epoch_time,
                    "best/val_mae": best_val_mae,
                    "epoch": epoch + 1,
                }, step=global_step)
            
            epoch_history.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_mse": val_metrics["mse"],
                "val_r2": val_metrics["r2"],
                "epoch_time_seconds": epoch_time,
            })

            if val_metrics['mae'] < best_val_mae:
                best_val_mae = val_metrics['mae']
                torch.save(model.state_dict(), save_path)
                print(f"*** New best model saved to {save_path} with MAE: {best_val_mae:.4f} ***")

            print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds ({epoch_time/60:.2f} minutes).")

        global_time = time.time() - global_start_time
        print(f"Training Complete! Total time: {global_time:.2f} seconds ({global_time/60:.2f} minutes / {global_time/3600:.2f} hours).")

        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if torch.cuda.is_available() else 0.0
        )

        # --- JSON Results Logging ---
        results = {
            "seed": args.seed,
            "model_type": args.model_type,
            "config": vars(args),
            "best_val_mae": float(best_val_mae),
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "total_time_seconds": global_time,
            "epochs": epoch_history,
        }
        json_path = Path(args.json_log_path.format(seed=args.seed, run_name=args.wandb_run_name, model=args.model_type))
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {json_path}")

    finally:
        if wandb is not None:
            wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QuakeWaveMamba2 for Seismic Regression")
    
    parser.add_argument("--arrow_paths", nargs='+', default=DEFAULT_ARROW_PATHS, help="List of paths to the cached .arrow dataset directories")
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH, help="Path to the merged augmented events CSV file")
    
    parser.add_argument("--num_workers", type=int, default=16, help="Number of dataloader workers (Maximized for H100)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (Maximized for 80GB VRAM in BF16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (Increased to compensate for large batch size)")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"], help="Optimizer type")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "step", "none"], help="Learning rate scheduler type")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="Number of warmup epochs for learning rate scheduler")
    parser.add_argument("--loss", type=str, default="mse", choices=["l1", "mse", "binned_mse"], help="Loss function to use") # TODO: Update Losses here
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    
    # Model type
    parser.add_argument("--model_type", type=str, choices=["mamba2", "bi_lstm", "transformer"], default="mamba2", help="Model architecture to use")

    # Mamba2 parameters
    parser.add_argument("--d_model", type=int, default=128, help="Model hidden dimension")
    parser.add_argument("--d_state", type=int, default=64, help="Mamba2 state dimension")
    parser.add_argument("--mamba_headdim", type=int, default=32, help="Mamba2 headdim. Set to 32 to ensure multiple-of-8 for causal_conv1d.")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of Mamba2 blocks")
    
    # LSTM parameters
    parser.add_argument("--lstm_hidden_size", type=int, default=128, help="Hidden size for LSTM model")
    parser.add_argument("--lstm_layers", type=int, default=2, help="Number of layers for LSTM model")
    parser.add_argument("--lstm_dropout", type=float, default=0.2, help="Dropout for LSTM model")
    
    # Transformer parameters
    parser.add_argument("--tf_nhead", type=int, default=8, help="Number of attention heads for Transformer model")
    parser.add_argument("--tf_layers", type=int, default=4, help="Number of encoder layers for Transformer model")
    parser.add_argument("--tf_dim_feedforward", type=int, default=512, help="Dimension of the feedforward network in Transformer model")
    parser.add_argument("--tf_dropout", type=float, default=0.2, help="Dropout for Transformer model")

    # Output arguments
    parser.add_argument("--save_path", type=str, default=str(DEFAULT_SAVE_PATH), help="Path to save the best model weights")
    parser.add_argument("--json_log_path", type=str, default=str(PROJECT_ROOT / "results" / "{model}_seed{seed}.json"), help="Path template for JSON results (supports {seed}, {run_name}, {model})")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="quake-wave-mamba2", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default="", help="W&B entity/team")
    parser.add_argument("--wandb_run_name", type=str, default="run-h100-bf16", help="W&B run name")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="offline",
        help="W&B mode: online, offline, or disabled",
    )

    args = parser.parse_args()
    
    train(args)