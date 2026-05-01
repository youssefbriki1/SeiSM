"""
Evaluate all three models (BiLSTM, Mamba2, Transformer) on the test set
using saved checkpoints, then plot per-magnitude-class MAE averaged over seeds.

No retraining required — uses best_<model>_l_seed<seed>.pth checkpoints.
"""
import argparse
import os
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset

from models import QuakeWaveMamba2, WaveformTransformer, BiWaveformLSTM

PROJECT_ROOT = Path(__file__).resolve().parents[1]

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})

SEEDS = [42, 1, 2, 3, 4]
BINS  = ["<1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7", ">=7"]

COLORS = {
    "bi_lstm":     "#1f77b4",
    "mamba2":      "#ff7f0e",
    "transformer": "#2ca02c",
}
DISPLAY_NAMES = {
    "bi_lstm":     "BiLSTM (RNN)",
    "mamba2":      "Mamba2 (SSM)",
    "transformer": "Transformer",
}

# All models were trained with default constructor args (waveform_train.py passes none)
MODEL_BUILDERS = {
    "mamba2":      lambda: QuakeWaveMamba2(),
    "transformer": lambda: WaveformTransformer(),
    "bi_lstm":     lambda: BiWaveformLSTM(),
}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-magnitude-class MAE evaluation using saved checkpoints."
    )
    parser.add_argument("--arrow_path", type=str,
        default=str(PROJECT_ROOT / "data/ceed_waveforms/AI4EPS___ceed/station_test"
                    "/1.1.0/c062e7b0694b5aba3f4b3b624a764e52ecffbf5260ebfc550e1256de763c6e03"))
    parser.add_argument("--csv_path", type=str,
        default=str(PROJECT_ROOT / "data/ceed_waveforms/events_test.csv"))
    parser.add_argument("--checkpoints_dir", type=str,
        default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--output_dir", type=str,
        default=str(PROJECT_ROOT / "results/analysis"))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--models", nargs="+",
        default=["bi_lstm", "mamba2", "transformer"],
        choices=list(MODEL_BUILDERS.keys()))
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset (mirrors evaluate_waveform.py)
# ──────────────────────────────────────────────────────────────────────────────

class SeismicDataset(torch.utils.data.Dataset):
    def __init__(self, arrow_dir: str, csv_path: str):
        arrow_files = sorted(glob.glob(os.path.join(arrow_dir, "*.arrow")))
        if not arrow_files:
            raise FileNotFoundError(f"No .arrow files found in {arrow_dir}")
        self.hf = load_dataset("arrow", data_files=arrow_files, split="train")
        df = pd.read_csv(csv_path)
        self.mag_lookup = dict(zip(df["event_time"].astype(str),
                                   df["magnitude"].fillna(0.0).astype(float)))
        self.hf = self.hf.with_format(type="torch", columns=["data"],
                                       output_all_columns=True)

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, idx):
        sample = self.hf[idx]
        wav = sample["data"]
        wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
        mag = torch.tensor([self.mag_lookup.get(str(sample["event_time"]), 0.0)],
                           dtype=torch.float32)
        mean = wav.mean(dim=1, keepdim=True)
        std  = wav.std(dim=1, keepdim=True) + 1e-5
        wav  = torch.nan_to_num((wav - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
        return wav.float(), mag


def get_bin(mag: float) -> str:
    if   mag < 1.0: return "<1"
    elif mag < 2.0: return "1-2"
    elif mag < 3.0: return "2-3"
    elif mag < 4.0: return "3-4"
    elif mag < 5.0: return "4-5"
    elif mag < 6.0: return "5-6"
    elif mag < 7.0: return "6-7"
    else:           return ">=7"


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model: torch.nn.Module, loader: DataLoader,
                  device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_true, all_pred = [], []
    for wavs, mags in tqdm(loader, desc="  inference", leave=False):
        wavs = wavs.to(device)
        if wavs.dim() == 2:
            wavs = wavs.unsqueeze(0)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = model(wavs)
        all_pred.extend(out.view(-1).float().cpu().numpy().tolist())
        all_true.extend(mags.view(-1).float().numpy().tolist())
    return np.array(all_true), np.array(all_pred)


def bin_mae(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    labels = [get_bin(t) for t in true]
    result = {}
    for b in BINS:
        idx = [i for i, lb in enumerate(labels) if lb == b]
        if idx:
            result[b] = float(np.mean(np.abs(true[idx] - pred[idx])))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_per_class_mae(
    per_seed_maes: dict[str, list[dict[str, float]]],
    bin_counts: dict[str, int],
    output_dir: Path,
) -> None:
    models = list(per_seed_maes.keys())
    valid_bins = [b for b in BINS if bin_counts.get(b, 0) > 0]
    x = np.arange(len(valid_bins))
    n_models = len(models)
    width = 0.7 / n_models
    offsets = [(i - (n_models - 1) / 2) * width for i in range(n_models)]

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, model in enumerate(models):
        seed_maes = per_seed_maes[model]  # list of dicts {bin: mae}
        means, stds = [], []
        for b in valid_bins:
            vals = [d[b] for d in seed_maes if b in d]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals)   if len(vals) > 1 else 0.0)

        means = np.array(means)
        stds  = np.array(stds)
        valid = ~np.isnan(means)

        ax.bar(x[valid] + offsets[i], means[valid], width,
               yerr=stds[valid], color=COLORS[model], edgecolor="black",
               alpha=0.82, capsize=5, label=DISPLAY_NAMES[model], zorder=2)

        # Individual seed dots
        for j, b in enumerate(valid_bins):
            vals = [d[b] for d in seed_maes if b in d]
            jitter = np.random.default_rng(i * 100 + j).uniform(
                -width * 0.25, width * 0.25, size=len(vals))
            ax.scatter(x[j] + offsets[i] + jitter, vals,
                       color="black", s=18, zorder=3, alpha=0.6)

    ax.set_title(
        "Per-Magnitude-Class MAE — Size L\n(mean ± std over 5 seeds, dots = individual seeds)",
        fontweight="bold",
    )
    ax.set_xlabel("Magnitude Class")
    ax.set_ylabel("Mean Absolute Error (MAE)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{b}\n(n={bin_counts.get(b, 0):,})" for b in valid_bins],
        fontsize=9,
    )
    ax.legend()
    plt.tight_layout()
    path = output_dir / "per_class_mae.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_overall_mae(
    per_seed_maes: dict[str, list[dict[str, float]]],
    per_seed_overall: dict[str, list[float]],
    output_dir: Path,
) -> None:
    models = list(per_seed_maes.keys())
    valid_bins = BINS  # will filter empty ones per model

    fig, axes = plt.subplots(1, len(models), figsize=(18, 5), sharey=False)
    fig.suptitle(
        "Per-Magnitude-Class MAE per Model — Size L (mean ± std over 5 seeds)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for ax, model in zip(axes, models):
        color = COLORS[model]
        seed_maes = per_seed_maes[model]
        bins_present = [b for b in BINS if any(b in d for d in seed_maes)]
        x = np.arange(len(bins_present))

        means, stds = [], []
        for b in bins_present:
            vals = [d[b] for d in seed_maes if b in d]
            means.append(np.mean(vals))
            stds.append(np.std(vals) if len(vals) > 1 else 0.0)

        ax.bar(x, means, yerr=stds, color=color, edgecolor="black",
               alpha=0.80, capsize=5, zorder=2)

        overall_mean = np.mean(per_seed_overall[model])
        overall_std  = np.std(per_seed_overall[model])
        ax.axhline(overall_mean, color=color, linestyle="--", linewidth=1.8,
                   label=f"Overall MAE: {overall_mean:.3f}±{overall_std:.3f}")

        ax.set_title(DISPLAY_NAMES[model], fontweight="bold", color=color)
        ax.set_xlabel("Magnitude Class")
        ax.set_ylabel("MAE")
        ax.set_xticks(x)
        ax.set_xticklabels(bins_present, rotation=30, ha="right", fontsize=9)
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = output_dir / "per_class_mae_panels.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoints_dir)

    # The test data lives in a dedicated station_test directory — use it all
    print("Loading dataset...")
    dataset = SeismicDataset(args.arrow_path, args.csv_path)
    print(f"Test set size: {len(dataset):,}")

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=True)

    # Compute bin counts from true labels (only needed once)
    # We'll collect them during the first model's first seed pass.
    bin_counts: dict[str, int] = {}
    true_ref: np.ndarray | None = None

    per_seed_maes:    dict[str, list[dict[str, float]]] = {m: [] for m in args.models}
    per_seed_overall: dict[str, list[float]]             = {m: [] for m in args.models}

    for model_key in args.models:
        print(f"\n{'='*60}")
        print(f"Model: {DISPLAY_NAMES[model_key]}")
        print(f"{'='*60}")

        for seed in SEEDS:
            ckpt_path = ckpt_dir / f"best_{model_key}_l_seed{seed}.pth"
            if not ckpt_path.exists():
                print(f"  [seed {seed}] Checkpoint not found: {ckpt_path} — skipping")
                continue

            print(f"  [seed {seed}] Loading {ckpt_path.name}")
            model = MODEL_BUILDERS[model_key]().to(device)
            ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state)

            true, pred = run_inference(model, loader, device)
            del model
            torch.cuda.empty_cache()

            if true_ref is None:
                true_ref = true
                labels = [get_bin(t) for t in true_ref]
                for b in BINS:
                    bin_counts[b] = sum(1 for lb in labels if lb == b)

            seed_bin_mae = bin_mae(true, pred)
            overall_mae  = float(np.mean(np.abs(true - pred)))

            per_seed_maes[model_key].append(seed_bin_mae)
            per_seed_overall[model_key].append(overall_mae)

            print(f"         overall MAE = {overall_mae:.4f}")

    # Print summary table
    print("\n" + "=" * 65)
    print("PER-CLASS MAE SUMMARY (mean ± std over seeds)")
    print("=" * 65)
    valid_bins = [b for b in BINS if bin_counts.get(b, 0) > 0]
    header = f"{'Bin':<6}  {'n':>6}" + "".join(
        f"  {DISPLAY_NAMES[m]:>18}" for m in args.models
    )
    print(header)
    print("-" * 65)
    for b in valid_bins:
        row = f"{b:<6}  {bin_counts[b]:>6}"
        for m in args.models:
            vals = [d[b] for d in per_seed_maes[m] if b in d]
            if vals:
                row += f"  {np.mean(vals):>8.4f}±{np.std(vals):<8.4f}"
            else:
                row += f"  {'N/A':>18}"
        print(row)
    print("-" * 65)
    row = f"{'Overall':<6}  {'':>6}"
    for m in args.models:
        vals = per_seed_overall[m]
        row += f"  {np.mean(vals):>8.4f}±{np.std(vals):<8.4f}"
    print(row)
    print("=" * 65)

    # Save CSV
    records = []
    for b in valid_bins:
        rec = {"bin": b, "n": bin_counts[b]}
        for m in args.models:
            vals = [d[b] for d in per_seed_maes[m] if b in d]
            rec[f"{m}_mean_mae"] = np.mean(vals) if vals else np.nan
            rec[f"{m}_std_mae"]  = np.std(vals)  if vals else np.nan
        records.append(rec)
    pd.DataFrame(records).to_csv(output_dir / "per_class_mae.csv", index=False)
    print(f"\nper_class_mae.csv saved to {output_dir}")

    # Plots
    plot_per_class_mae(per_seed_maes, bin_counts, output_dir)
    plot_overall_mae(per_seed_maes, per_seed_overall, output_dir)


if __name__ == "__main__":
    main()
