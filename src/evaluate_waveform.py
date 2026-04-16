import os
import glob
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset

# Import both models
from models import QuakeWaveMamba2, WaveformTransformer

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

def get_bin(magnitude):
    if magnitude < 1.0: return "<1"
    elif 1.0 <= magnitude < 2.0: return "1-2"
    elif 2.0 <= magnitude < 3.0: return "2-3"
    elif 3.0 <= magnitude < 4.0: return "3-4"
    elif 4.0 <= magnitude < 5.0: return "4-5"
    elif 5.0 <= magnitude < 6.0: return "5-6"
    elif 6.0 <= magnitude < 7.0: return "6-7"
    else: return ">=7"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Mamba2 and Transformer Models by Magnitude Bins")
    
    # Dataset args
    parser.add_argument("--arrow_paths", nargs='+', 
                        default=[
                            "/scratch/brikiyou/ift3710/data/ceed_waveforms/AI4EPS___ceed/station_test/1.1.0/c062e7b0694b5aba3f4b3b624a764e52ecffbf5260ebfc550e1256de763c6e03",
                            "/scratch/brikiyou/ift3710/data/ceed_waveforms/AI4EPS___ceed/station_test/1.1.0/augmented_data"
                        ],
                        help="Paths to huggingface arrow file directories")
    parser.add_argument("--csv_path", type=str, 
                        default="/scratch/brikiyou/ift3710/data/ceed_waveforms/events_test_augmented.csv",
                        help="Path to CSV containing event magnitudes")
    
    # Checkpoint paths
    parser.add_argument("--mamba_checkpoint", type=str,
                        default="/scratch/brikiyou/ift3710/checkpoints/waveforms/best_mamba2_adamw_waveform.pth",
                        help="Path to the Mamba2 model checkpoint")
    parser.add_argument("--transformer_checkpoint", type=str,
                        default="/scratch/brikiyou/ift3710/checkpoints/waveforms/best_transformer_adamw_waveform.pth",
                        help="Path to the Transformer model checkpoint")
    
    # General Data Loading
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    
    return parser.parse_args()

def evaluate():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset setup
    from torch.utils.data import ConcatDataset, DataLoader
    csv_path = str(args.csv_path)
    
    individual_datasets = []
    for path in args.arrow_paths:
        ds = ArrowSeismicDataset(str(path), csv_path)
        individual_datasets.append(ds)
        
    print("Stitching datasets together via PyTorch ConcatDataset...")
    full_dataset = ConcatDataset(individual_datasets)
    print(f"Total Combined Dataset Size: {len(full_dataset)}")
    
    train_size = int(0.6 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    _, _, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 2. Models Configuration
    models_config = {
        "Mamba2 (SSM)": {
            "checkpoint": args.mamba_checkpoint,
            "model": QuakeWaveMamba2(in_channels=3, d_model=128, d_state=64, n_layers=4, headdim=32),
            "color": "#ff7f0e" # Orange
        },
        "Transformer": {
            "checkpoint": args.transformer_checkpoint,
            "model": WaveformTransformer(in_channels=3, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.2, output_size=1),
            "color": "#2ca02c" # Green
        }
    }

    results = {}

    # 3. Inference Loop for Both Models
    for name, config in models_config.items():
        print(f"\n[{name}] Loading model from {config['checkpoint']}...")
        model = config["model"].to(device)
        
        # Load checkpoint securely handling different save formats
        checkpoint = torch.load(config["checkpoint"], map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()

        all_true = []
        all_pred = []
        
        with torch.no_grad():
            for waveforms, magnitudes in tqdm(test_loader, desc=f"Evaluating {name}"):
                waveforms = waveforms.to(device)
                if waveforms.dim() == 2:
                    waveforms = waveforms.unsqueeze(0)
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(waveforms)
                
                outputs = outputs.view(-1).float()
                magnitudes = magnitudes.view(-1).float()
                
                all_true.extend(magnitudes.cpu().numpy().tolist())
                all_pred.extend(outputs.cpu().numpy().tolist())
                
        results[name] = {
            "true": np.array(all_true),
            "pred": np.array(all_pred)
        }

    # 4. Process into Bins
    bins = ["<1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7", ">=7"]
    
    bin_counts = {b: 0 for b in bins}
    model_bin_maes = {name: {b: 0.0 for b in bins} for name in models_config.keys()}
    
    # True values are identical across models, use the first one's true values for bin assignment
    first_model_name = list(models_config.keys())[0]
    all_true_shared = results[first_model_name]["true"]
    true_bins = [get_bin(t) for t in all_true_shared]
    
    for b in bins:
        b_idx = [i for i, tb in enumerate(true_bins) if tb == b]
        bin_counts[b] = len(b_idx)
        
        for name in models_config.keys():
            if len(b_idx) > 0:
                t_arr = results[name]["true"][b_idx]
                p_arr = results[name]["pred"][b_idx]
                model_bin_maes[name][b] = np.mean(np.abs(t_arr - p_arr))
            else:
                model_bin_maes[name][b] = 0.0

    # Print MAEs per magnitude class
    print("\n" + "="*60)
    print("MAE RESULTS BY MAGNITUDE CLASS")
    print("="*60)
    for name in models_config.keys():
        print(f"\n--- {name} ---")
        overall_mae = np.mean(np.abs(results[name]["true"] - results[name]["pred"]))
        print(f"Overall MAE: {overall_mae:.4f}")
        for b in bins:
            if bin_counts[b] > 0:
                print(f"Bin {b:<4} (n={bin_counts[b]:<5}): MAE = {model_bin_maes[name][b]:.4f}")
    print("="*60 + "\n")

    # 5. Generate Visualizations
    out_dir = os.path.join(os.path.dirname(args.mamba_checkpoint), "..", "plots")
    os.makedirs(out_dir, exist_ok=True)
    print("Generating comparative visualizations...")

    # Plot 1: Grouped Bar Chart of MAE per Bin
    fig, ax = plt.subplots(figsize=(14, 7))
    
    valid_bins = [b for b in bins if bin_counts[b] > 0]
    x = np.arange(len(valid_bins))
    width = 0.35  # Adjusted for 2 models
    
    for i, (name, config) in enumerate(models_config.items()):
        maes = [model_bin_maes[name][b] for b in valid_bins]
        offset = width/2 if i == 1 else -width/2
        ax.bar(x + offset, maes, width, label=name, color=config["color"], edgecolor='black', alpha=0.8)
        
    ax.set_title("Mean Absolute Error (MAE) by Magnitude Bin Comparison", fontsize=14, fontweight='bold')
    ax.set_xlabel("Magnitude Range", fontsize=12)
    ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={bin_counts[b]})" for b in valid_bins])
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    bar_path = os.path.join(out_dir, "mamba_vs_transformer_mae_bins.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    
    # Plot 2: Scatter plot True vs Predicted (Side-by-side)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    
    if len(all_true_shared) > 10000:
        idx = np.random.choice(len(all_true_shared), 10000, replace=False)
    else:
        idx = np.arange(len(all_true_shared))
        
    for i, (name, config) in enumerate(models_config.items()):
        ax = axes[i]
        scatter_t = results[name]["true"][idx]
        scatter_p = results[name]["pred"][idx]
        
        ax.scatter(scatter_t, scatter_p, alpha=0.2, s=5, color=config["color"])
        
        min_val = min(min(scatter_t), min(scatter_p))
        max_val = max(max(scatter_t), max(scatter_p))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
        
        mae_overall = np.mean(np.abs(results[name]["true"] - results[name]["pred"]))
        ax.set_title(f"{name}\nOverall MAE: {mae_overall:.4f}", fontsize=13)
        ax.set_xlabel("True Magnitude")
        if i == 0:
            ax.set_ylabel("Predicted Magnitude")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        
    plt.tight_layout()
    scatter_path = os.path.join(out_dir, "mamba_vs_transformer_scatter.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {out_dir}")

if __name__ == "__main__":
    evaluate()