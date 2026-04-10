import os
import glob
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from models.quakewave_mamba import QuakeWaveMamba2

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
    if magnitude < 1.0:
        return "<1"
    elif 1.0 <= magnitude < 2.0:
        return "1-2"
    elif 2.0 <= magnitude < 3.0:
        return "2-3"
    elif 3.0 <= magnitude < 4.0:
        return "3-4"
    elif 4.0 <= magnitude < 5.0:
        return "4-5"
    elif 5.0 <= magnitude < 6.0:
        return "5-6"
    elif 6.0 <= magnitude < 7.0:
        return "6-7"
    else:
        return ">=7"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SSM Waveform Model by Magnitude Bins")
    
    parser.add_argument("--arrow_path", type=str, 
                        default="/scratch/brikiyou/ift3710/data/ceed_waveforms/AI4EPS___ceed/station_test/1.1.0/c062e7b0694b5aba3f4b3b624a764e52ecffbf5260ebfc550e1256de763c6e03",
                        help="Path to huggingface arrow file directory")
    parser.add_argument("--csv_path", type=str, 
                        default="/scratch/brikiyou/ift3710/data/ceed_waveforms/events_test_augmented.csv",
                        help="Path to CSV containing event magnitudes")
    parser.add_argument("--checkpoint", type=str,
                        default="/scratch/brikiyou/ift3710/checkpoints/best_quake_mamba2_waveform.pth",
                        help="Path to the model checkpoint")
    
    # Model configuration (must match training parameters)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_state", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--mamba_headdim", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    
    return parser.parse_args()

def evaluate():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset setup
    arrow_path = str(args.arrow_path)
    csv_path = str(args.csv_path)
    full_dataset = ArrowSeismicDataset(arrow_path, csv_path)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 2. Model setup
    print(f"Loading model from {args.checkpoint}...")
    model = QuakeWaveMamba2(
        in_channels=3,
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        headdim=args.mamba_headdim,
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Handle both bare state_dict and dicts containing 'model_state_dict'
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Inference
    all_true = []
    all_pred = []
    
    print("Running inference on validation set...")
    with torch.no_grad():
        for waveforms, magnitudes in tqdm(val_loader, desc="Evaluating"):
            waveforms = waveforms.to(device)
            # Add dummy batch dim if shape is (C, L)
            if waveforms.dim() == 2:
                waveforms = waveforms.unsqueeze(0)
            
            # Forward pass
            outputs = model(waveforms)
            
            # Squeeze outputs and targets if they are (B, 1)
            outputs = outputs.view(-1)
            magnitudes = magnitudes.view(-1)
            
            all_true.extend(magnitudes.cpu().numpy().tolist())
            all_pred.extend(outputs.cpu().numpy().tolist())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    
    # 4. Binning logic
    bins = ["<1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7", ">=7"]
    bin_true = {b: [] for b in bins}
    bin_pred = {b: [] for b in bins}
    
    for t, p in zip(all_true, all_pred):
        b = get_bin(t)
        bin_true[b].append(t)
        bin_pred[b].append(p)
        
    # Calculate MAE per bin
    print("\n--- Evaluation by Magnitude Bin ---")
    print(f"{'Magnitude Bin':<15} | {'Count':<10} | {'MAE':<10}")
    print("-" * 40)
    
    bin_maes = []
    counts = []
    
    for b in bins:
        t_arr = np.array(bin_true[b])
        p_arr = np.array(bin_pred[b])
        
        if len(t_arr) > 0:
            mae = np.mean(np.abs(t_arr - p_arr))
        else:
            mae = 0.0
            
        bin_maes.append(mae)
        counts.append(len(t_arr))
        
        mae_str = f"{mae:.4f}" if len(t_arr) > 0 else "N/A"
        print(f"{b:<15} | {len(t_arr):<10} | {mae_str:<10}")
        
    overall_mae = np.mean(np.abs(all_true - all_pred))
    print("-" * 40)
    print(f"{'Overall':<15} | {len(all_true):<10} | {overall_mae:.4f}")
    
    # 5. Visualization
    print("\nGenerating visualizations...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: MAE per Bin
    # Filter out bins with 0 samples for the plot
    valid_indices = [i for i, c in enumerate(counts) if c > 0]
    valid_bins = [bins[i] for i in valid_indices]
    valid_maes = [bin_maes[i] for i in valid_indices]
    
    ax1.bar(valid_bins, valid_maes, color='skyblue', edgecolor='black')
    ax1.set_title("Mean Absolute Error (MAE) by Magnitude Bin")
    ax1.set_xlabel("Magnitude Range")
    ax1.set_ylabel("MAE")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate bars with counts
    for i, v in enumerate(valid_maes):
        ax1.text(i, v + 0.01, f"n={counts[valid_indices[i]]}", ha='center', va='bottom', fontsize=9)
        
    # Plot 2: Scatter plot True vs Predicted
    # Subsample scatter plot if there are too many points to avoid massive files
    if len(all_true) > 10000:
        idx = np.random.choice(len(all_true), 10000, replace=False)
        scatter_t = all_true[idx]
        scatter_p = all_pred[idx]
    else:
        scatter_t = all_true
        scatter_p = all_pred
        
    ax2.scatter(scatter_t, scatter_p, alpha=0.2, s=5, c='coral')
    
    # Add identity line
    min_val = min(min(scatter_t), min(scatter_p))
    max_val = max(max(scatter_t), max(scatter_p))
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    
    ax2.set_title("True vs Predicted Magnitude")
    ax2.set_xlabel("True Magnitude")
    ax2.set_ylabel("Predicted Magnitude")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(args.checkpoint), "..", "plots/magnitude_evaluation.png")
    plt.savefig(output_path, dpi=300)
    print(f"Visualizations saved to {output_path}")

if __name__ == "__main__":
    evaluate()
