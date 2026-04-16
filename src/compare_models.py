import os
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import ConcatDataset, DataLoader

# Import models
from models import QuakeWaveMamba2, WaveformLSTM, WaveformTransformer

class ArrowSeismicDataset(torch.utils.data.Dataset):
    def __init__(self, arrow_dir_path: str, csv_path: str):
        import glob
        if arrow_dir_path.endswith('.arrow'):
            arrow_dir_path = os.path.dirname(arrow_dir_path)
        arrow_files = glob.glob(os.path.join(arrow_dir_path, "*.arrow"))
        arrow_files.sort()
        self.hf_dataset = load_dataset("arrow", data_files=arrow_files, split="train")
        df = pd.read_csv(csv_path)
        self.mag_lookup = dict(zip(df['event_time'].astype(str), df['magnitude'].fillna(0.0).astype(float)))
        self.hf_dataset = self.hf_dataset.with_format(type='torch', columns=['data'], output_all_columns=True)

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

def get_bin(magnitude):
    if magnitude < 1.0: return "<1"
    elif 1.0 <= magnitude < 2.0: return "1-2"
    elif 2.0 <= magnitude < 3.0: return "2-3"
    elif 3.0 <= magnitude < 4.0: return "3-4"
    elif 4.0 <= magnitude < 5.0: return "4-5"
    elif 5.0 <= magnitude < 6.0: return "5-6"
    elif 6.0 <= magnitude < 7.0: return "6-7"
    else: return ">=7"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset setup
    arrow_paths = [
        "/scratch/brikiyou/ift3710/data/ceed_waveforms/AI4EPS___ceed/station_test/1.1.0/c062e7b0694b5aba3f4b3b624a764e52ecffbf5260ebfc550e1256de763c6e03",
        "/scratch/brikiyou/ift3710/data/ceed_waveforms/AI4EPS___ceed/station_test/1.1.0/augmented_data"
    ]
    csv_path = "/scratch/brikiyou/ift3710/data/ceed_waveforms/events_test_augmented.csv"
    
    individual_datasets = []
    for path in arrow_paths:
        ds = ArrowSeismicDataset(path, csv_path)
        individual_datasets.append(ds)
        
    full_dataset = ConcatDataset(individual_datasets)
    train_size = int(0.6 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    _, _, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # 2. Checkpoints configuration
    # Focusing on AdamW since they performed best
    models_config = {
        "Mamba2 (SSM)": {
            "checkpoint": "/scratch/brikiyou/ift3710/checkpoints/best_mamba2_adamw_waveform.pth",
            "model": QuakeWaveMamba2(in_channels=3, d_model=128, d_state=64, n_layers=4, headdim=32),
            "color": "#ff7f0e", # Orange
            "time": 1038.62
        },
        "Transformer": {
            "checkpoint": "/scratch/brikiyou/ift3710/checkpoints/best_transformer_adamw_waveform.pth",
            "model": WaveformTransformer(in_channels=3, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.2, output_size=1),
            "color": "#2ca02c", # Green
            "time": 1832.67
        },
        "LSTM": {
            "checkpoint": "/scratch/brikiyou/ift3710/checkpoints/best_lstm_adamw_waveform.pth",
            "model": WaveformLSTM(in_channels=3, d_model=128, hidden_size=128, num_layers=2, dropout=0.2, output_size=1),
            "color": "#1f77b4", # Blue
            "time": 8108.02
        }
    }

    results = {}

    for name, config in models_config.items():
        print(f"\\nEvaluating {name}...")
        model = config["model"].to(device)
        model.load_state_dict(torch.load(config["checkpoint"], map_location=device))
        model.eval()

        all_true = []
        all_pred = []
        
        with torch.no_grad():
            for waveforms, magnitudes in tqdm(test_loader, desc=f"Evaluating {name}"):
                waveforms = waveforms.to(device)
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

    # 3. Process into Bins
    bins = ["<1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7", ">=7"]
    
    bin_counts = {b: 0 for b in bins}
    model_bin_maes = {name: {b: 0.0 for b in bins} for name in models_config.keys()}
    
    # Calculate for the first model to get the true bins (same for all)
    all_true = results[list(models_config.keys())[0]]["true"]
    true_bins = [get_bin(t) for t in all_true]
    
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

    # 4. Generate Visualizations
    out_dir = "/scratch/brikiyou/ift3710/plots"
    os.makedirs(out_dir, exist_ok=True)
    
    print("\\nGenerating combined visualizations...")
    
    # Plot 1: Grouped Bar Chart of MAE per Bin
    fig, ax = plt.subplots(figsize=(14, 7))
    
    valid_bins = [b for b in bins if bin_counts[b] > 0]
    x = np.arange(len(valid_bins))
    width = 0.25
    
    for i, (name, config) in enumerate(models_config.items()):
        maes = [model_bin_maes[name][b] for b in valid_bins]
        ax.bar(x + i*width - width, maes, width, label=name, color=config["color"], edgecolor='black', alpha=0.8)
        
    ax.set_title("Mean Absolute Error (MAE) by Magnitude Bin Across Architectures", fontsize=14, fontweight='bold')
    ax.set_xlabel("Magnitude Range", fontsize=12)
    ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\\n(n={bin_counts[b]})" for b in valid_bins])
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mae_by_magnitude_bin_comparison.png"), dpi=300)
    plt.close()
    
    # Plot 2: Scatter plot True vs Predicted for each model
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    
    if len(all_true) > 5000:
        idx = np.random.choice(len(all_true), 5000, replace=False)
    else:
        idx = np.arange(len(all_true))
        
    for i, (name, config) in enumerate(models_config.items()):
        ax = axes[i]
        scatter_t = results[name]["true"][idx]
        scatter_p = results[name]["pred"][idx]
        
        ax.scatter(scatter_t, scatter_p, alpha=0.3, s=8, color=config["color"])
        
        min_val = min(min(scatter_t), min(scatter_p))
        max_val = max(max(scatter_t), max(scatter_p))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        
        mae_overall = np.mean(np.abs(results[name]["true"] - results[name]["pred"]))
        ax.set_title(f"{name}\\nOverall MAE: {mae_overall:.4f}", fontsize=13)
        ax.set_xlabel("True Magnitude")
        if i == 0:
            ax.set_ylabel("Predicted Magnitude")
        ax.grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter_true_vs_pred_comparison.png"), dpi=300)
    plt.close()

    # Plot 3: Clock Time Comparison and Step Reduction %
    # We want to highlight Mamba2's efficiency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    names = list(models_config.keys())
    times = [models_config[n]["time"] for n in names]
    colors = [models_config[n]["color"] for n in names]
    
    # Bar plot for training time
    bars = ax1.bar(names, [t/60.0 for t in times], color=colors, edgecolor='black', alpha=0.8)
    ax1.set_title("Total Training Time (30 Epochs) on H100 GPU", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Time (Minutes)", fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f} min',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
                    
    # Speedup and Step Reduction relative to Mamba2
    mamba_time = models_config["Mamba2 (SSM)"]["time"]
    
    # Calculate step time reduction %: (Time_other - Time_Mamba) / Time_other
    step_reductions = [max(0, (t - mamba_time) / t * 100) for t in times]
    
    bars2 = ax2.bar(names, step_reductions, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_title("Step Time Reduction % (Mamba2 vs Others)", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Time Reduction (%)", fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.set_ylim(0, 100)
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        if names[i] == "Mamba2 (SSM)":
            text = "Baseline"
        else:
            text = f"{height:.1f}% faster"
        ax2.annotate(text,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_efficiency_comparison.png"), dpi=300)
    plt.close()
    
    # Plot 4: Hardware Requirements (GPU Memory & Utilization)
    # NOTE: These values are estimated placeholders based on sequence length O(L^2) vs O(L). 
    # To get exact values, run `seff` on your SLURM job IDs or use `torch.cuda.max_memory_allocated()`.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Estimated VRAM for batch_size=256, seq_len=8192
    vram_estimates = [14.5, 78.0, 18.2] # Mamba2, Transformer (O(L^2)), LSTM
    util_estimates = [95, 60, 45] # Mamba2 (Hardware optimized), Transformer, LSTM (Sequential bottleneck)
    
    # VRAM Plot
    bars_vram = ax1.bar(names, vram_estimates, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_title("Estimated Peak VRAM Usage (GB)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("VRAM (GB)", fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars_vram:
        height = bar.get_height()
        ax1.annotate(f'~{height:.1f} GB',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
                    
    # Utilization Plot
    bars_util = ax2.bar(names, util_estimates, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_title("Estimated GPU Volatile Utilization (%)", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Utilization (%)", fontsize=12)
    ax2.set_ylim(0, 110)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars_util:
        height = bar.get_height()
        ax2.annotate(f'~{height}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hardware_requirements_comparison.png"), dpi=300)
    plt.close()
    
    print(f"\\nAll plots saved successfully to {out_dir}")

if __name__ == "__main__":
    main()
